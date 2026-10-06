from __future__ import annotations

import argparse
import json
import os
import re
import time

from difflib import SequenceMatcher
from getpass import getpass
from typing import Any

import requests
from fastmcp import FastMCP


# ============================================================
# TMC REGULATORY SEARCH MCP
#
# Purpose:
#
#   Expose exactly one read-only MCP tool:
#
#       search_regulatory_content
#
# Runtime path:
#
#   TrueFoundry Agent
#           ↓
#   TrueFoundry MCP Gateway
#           ↓
#   Hosted STDIO MCP
#           ↓
#   search_regulatory_content
#           ↓
#   Entra Service Principal
#           ↓
#   Azure AI Search
#           ↓
#   regulatory-rag-chunks-index
#
# This server performs no writes.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TENANT_ID = (
    "2da0b43a-b8f0-45ed-8716-4fdeecec40fe"
)

DEFAULT_CLIENT_ID = (
    "d4ce6e46-7d67-4458-b5e3-61516f832610"
)

DEFAULT_SEARCH_ENDPOINT = (
    "https://vc-rpa-prod-eastus.search.windows.net"
)

DEFAULT_INDEX_NAME = (
    "regulatory-rag-chunks-index"
)

DEFAULT_API_VERSION = (
    "2026-04-01"
)

DEFAULT_SEMANTIC_CONFIGURATION = (
    "regulatory-semantic-config"
)


TENANT_ID = os.getenv(
    "AZURE_TENANT_ID",
    DEFAULT_TENANT_ID,
)

CLIENT_ID = os.getenv(
    "AZURE_CLIENT_ID",
    DEFAULT_CLIENT_ID,
)

SEARCH_ENDPOINT = os.getenv(
    "AZURE_SEARCH_ENDPOINT",
    DEFAULT_SEARCH_ENDPOINT,
).rstrip("/")

INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX",
    DEFAULT_INDEX_NAME,
)

API_VERSION = os.getenv(
    "AZURE_SEARCH_API_VERSION",
    DEFAULT_API_VERSION,
)

SEMANTIC_CONFIGURATION = os.getenv(
    "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
    DEFAULT_SEMANTIC_CONFIGURATION,
)


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    "TMC Regulatory Search"
)


# ============================================================
# TOKEN CACHE
# ============================================================

_token_cache: dict[str, Any] = {
    "access_token": None,
    "expires_at": 0,
}


# ============================================================
# AUTHENTICATION
# ============================================================

def _get_client_secret() -> str:
    """
    Retrieve the Azure client secret from the environment.

    In TrueFoundry this should be supplied through its
    credential / environment configuration.
    """

    secret = os.getenv(
        "AZURE_CLIENT_SECRET",
        "",
    ).strip()

    if not secret:
        raise RuntimeError(
            "AZURE_CLIENT_SECRET is not configured."
        )

    return secret


def _get_access_token() -> str:
    """
    Get or reuse an Entra access token for Azure AI Search.
    """

    now = time.time()

    cached_token = _token_cache.get(
        "access_token"
    )

    expires_at = _token_cache.get(
        "expires_at",
        0,
    )

    if (
        cached_token
        and expires_at > now + 60
    ):
        return cached_token

    client_secret = _get_client_secret()

    token_url = (
        "https://login.microsoftonline.com/"
        f"{TENANT_ID}/oauth2/v2.0/token"
    )

    response = requests.post(
        token_url,
        data={
            "client_id":
                CLIENT_ID,

            "client_secret":
                client_secret,

            "scope":
                "https://search.azure.com/.default",

            "grant_type":
                "client_credentials",
        },
        timeout=30,
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Unable to acquire Azure AI Search token. "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    body = response.json()

    access_token = body[
        "access_token"
    ]

    expires_in = int(
        body.get(
            "expires_in",
            3600,
        )
    )

    _token_cache[
        "access_token"
    ] = access_token

    _token_cache[
        "expires_at"
    ] = (
        now + expires_in
    )

    return access_token


# ============================================================
# CONTENT PARSING
# ============================================================

def _first_nonempty(
    *values: Any,
) -> Any:
    """
    Return the first useful non-empty value.
    """

    for value in values:

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if value:
                return value

            continue

        return value

    return None


def _parse_content_blob(
    raw_content: Any,
) -> dict[str, Any]:
    """
    Some Azure Search documents contain their source chunk
    serialized as JSON inside the content field.

    Parse that structure when possible.
    """

    if isinstance(
        raw_content,
        dict,
    ):
        return raw_content

    if not isinstance(
        raw_content,
        str,
    ):
        return {}

    text = raw_content.strip()

    if not text:
        return {}

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    return {}


def _lookup(
    search_result: dict[str, Any],
    parsed_content: dict[str, Any],
    key: str,
) -> Any:
    """
    Prefer the top-level Azure Search field.

    Fall back to metadata embedded inside content.
    """

    return _first_nonempty(
        search_result.get(
            key
        ),
        parsed_content.get(
            key
        ),
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def _looks_like_serialized_json(
    text: Any,
) -> bool:
    """
    Detect evidence strings that appear to be serialized JSON
    instead of regulatory prose.
    """

    if not isinstance(
        text,
        str,
    ):
        return False

    stripped = text.strip()

    if not stripped:
        return False

    if (
        stripped.startswith("{")
        or stripped.startswith("[")
    ):
        return True

    beginning = stripped[
        :500
    ].lower()

    json_markers = [
        '"document_id"',
        '"parent_document_id"',
        '"document_title"',
        '"source_category"',
        '"document_scope"',
        '"source_authority"',
    ]

    matches = sum(
        1
        for marker
        in json_markers
        if marker in beginning
    )

    return matches >= 2


def _clean_text(
    text: Any,
    max_length: int = 2500,
) -> str:
    """
    Normalize regulatory evidence for model consumption.
    """

    if not isinstance(
        text,
        str,
    ):
        return ""

    text = text.strip()

    if not text:
        return ""

    text = re.sub(
        r"-{2,}\s*Page\s+(\d+)\s*-{2,}",
        r"[Page \1]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip()

    if len(text) > max_length:

        text = (
            text[:max_length].rstrip()
            + "..."
        )

    return text


def _extract_caption(
    search_result: dict[str, Any],
) -> str | None:
    """
    Retrieve the Azure semantic caption when it contains
    usable regulatory prose.
    """

    captions = search_result.get(
        "@search.captions",
        [],
    )

    if not captions:
        return None

    first_caption = captions[0]

    if not isinstance(
        first_caption,
        dict,
    ):
        return None

    text = first_caption.get(
        "text"
    )

    if not text:
        return None

    if _looks_like_serialized_json(
        text
    ):
        return None

    cleaned = _clean_text(
        text
    )

    return (
        cleaned
        if cleaned
        else None
    )


def _extract_passage(
    search_result: dict[str, Any],
    parsed_content: dict[str, Any],
) -> str:
    """
    Return the cleanest available regulatory passage.

    Priority:

    1. Semantic caption
    2. Parsed source content
    3. Raw content when it is not serialized JSON
    """

    caption = _extract_caption(
        search_result
    )

    if caption:
        return caption


    nested_content = (
        parsed_content.get(
            "content"
        )
    )

    if (
        isinstance(
            nested_content,
            str,
        )
        and
        not _looks_like_serialized_json(
            nested_content
        )
    ):

        cleaned = _clean_text(
            nested_content
        )

        if cleaned:
            return cleaned


    raw_content = search_result.get(
        "content"
    )

    if (
        isinstance(
            raw_content,
            str,
        )
        and
        not _looks_like_serialized_json(
            raw_content
        )
    ):

        cleaned = _clean_text(
            raw_content
        )

        if cleaned:
            return cleaned


    return ""


# ============================================================
# DOCUMENT FILTERING
# ============================================================

def _is_excluded(
    search_result: dict[str, Any],
    parsed_content: dict[str, Any],
) -> tuple[bool, str | None]:
    """
    Reject crawler/error documents that should not be used as
    regulatory evidence.
    """

    document_scope = str(
        _lookup(
            search_result,
            parsed_content,
            "document_scope",
        )
        or ""
    ).strip().lower()

    source_category = str(
        _lookup(
            search_result,
            parsed_content,
            "source_category",
        )
        or ""
    ).strip().lower()


    if document_scope == "exclude":

        return (
            True,
            "document_scope=exclude",
        )


    if source_category in {
        "error_or_blocked",
        "error_page",
    }:

        return (
            True,
            (
                "source_category="
                f"{source_category}"
            ),
        )


    return (
        False,
        None,
    )


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def _normalize_for_comparison(
    text: str,
) -> str:
    """
    Normalize evidence for duplicate comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _passages_are_duplicates(
    first: str,
    second: str,
) -> bool:
    """
    Determine whether two evidence passages are substantially
    the same.
    """

    first_normalized = (
        _normalize_for_comparison(
            first
        )
    )

    second_normalized = (
        _normalize_for_comparison(
            second
        )
    )

    if (
        not first_normalized
        or not second_normalized
    ):
        return False


    if (
        first_normalized
        == second_normalized
    ):
        return True


    prefix_length = min(
        350,
        len(
            first_normalized
        ),
        len(
            second_normalized
        ),
    )

    if prefix_length >= 150:

        if (
            first_normalized[
                :prefix_length
            ]
            ==
            second_normalized[
                :prefix_length
            ]
        ):
            return True


    comparison_first = (
        first_normalized[
            :1800
        ]
    )

    comparison_second = (
        second_normalized[
            :1800
        ]
    )

    similarity = SequenceMatcher(
        None,
        comparison_first,
        comparison_second,
    ).ratio()

    return similarity >= 0.90


def _find_duplicate_rank(
    passage: str,
    evidence: list[
        dict[str, Any]
    ],
) -> int | None:
    """
    Return the rank of matching evidence already accepted.
    """

    for item in evidence:

        existing_passage = (
            item.get(
                "passage"
            )
            or ""
        )

        if _passages_are_duplicates(
            passage,
            existing_passage,
        ):

            return item.get(
                "rank"
            )

    return None


# ============================================================
# AZURE AI SEARCH
# ============================================================

def _search_azure_ai_search(
    query: str,
    requested_top: int,
) -> dict[str, Any]:
    """
    Execute semantic search against the TMC regulatory index.
    """

    access_token = (
        _get_access_token()
    )

    search_url = (
        f"{SEARCH_ENDPOINT}/"
        f"indexes/{INDEX_NAME}/docs/search"
        f"?api-version={API_VERSION}"
    )

    fetch_top = min(
        max(
            requested_top * 5,
            20,
        ),
        50,
    )

    payload = {
        "search":
            query,

        "queryType":
            "semantic",

        "semanticConfiguration":
            SEMANTIC_CONFIGURATION,

        "top":
            fetch_top,

        "select": (
            "chunk_id,"
            "document_id,"
            "document_title,"
            "title,"
            "section_id,"
            "section_title,"
            "section_heading,"
            "source_category,"
            "document_scope,"
            "source_authority,"
            "issuing_agency,"
            "jurisdiction,"
            "state,"
            "discipline,"
            "provider_type,"
            "version,"
            "regulatory_status,"
            "effective_date,"
            "official_source_url,"
            "canonical_source_url,"
            "source_url,"
            "content"
        ),

        "captions":
            "extractive",

        "answers":
            "extractive|count-3",
    }

    response = requests.post(
        search_url,
        headers={
            "Authorization":
                f"Bearer {access_token}",

            "Content-Type":
                "application/json",
        },
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Azure AI Search query failed. "
            f"HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


# ============================================================
# CORE SEARCH IMPLEMENTATION
# ============================================================

def search_regulatory_content_impl(
    query: str,
    top: int = 5,
) -> dict[str, Any]:
    """
    Execute clean, read-only regulatory semantic search.
    """

    query = (
        query
        if isinstance(
            query,
            str,
        )
        else ""
    ).strip()


    if not query:

        raise ValueError(
            "query is required."
        )


    if len(query) > 4000:

        raise ValueError(
            "query must be 4000 characters or fewer."
        )


    try:

        top = int(
            top
        )

    except Exception as exc:

        raise ValueError(
            "top must be an integer."
        ) from exc


    top = max(
        1,
        min(
            top,
            10,
        ),
    )


    raw_body = (
        _search_azure_ai_search(
            query=query,
            requested_top=top,
        )
    )

    raw_results = raw_body.get(
        "value",
        [],
    )


    evidence: list[
        dict[str, Any]
    ] = []

    excluded_results: list[
        dict[str, Any]
    ] = []

    duplicate_results: list[
        dict[str, Any]
    ] = []

    empty_passage_results: list[
        dict[str, Any]
    ] = []

    json_passage_results: list[
        dict[str, Any]
    ] = []


    for result in raw_results:

        if not isinstance(
            result,
            dict,
        ):
            continue


        parsed_content = (
            _parse_content_blob(
                result.get(
                    "content"
                )
            )
        )


        excluded, reason = (
            _is_excluded(
                result,
                parsed_content,
            )
        )

        if excluded:

            excluded_results.append(
                {
                    "chunk_id":
                        _lookup(
                            result,
                            parsed_content,
                            "chunk_id",
                        ),

                    "reason":
                        reason,
                }
            )

            continue


        passage = (
            _extract_passage(
                result,
                parsed_content,
            )
        )


        if not passage:

            empty_passage_results.append(
                {
                    "chunk_id":
                        _lookup(
                            result,
                            parsed_content,
                            "chunk_id",
                        )
                }
            )

            continue


        if _looks_like_serialized_json(
            passage
        ):

            json_passage_results.append(
                {
                    "chunk_id":
                        _lookup(
                            result,
                            parsed_content,
                            "chunk_id",
                        )
                }
            )

            continue


        duplicate_rank = (
            _find_duplicate_rank(
                passage,
                evidence,
            )
        )

        if duplicate_rank is not None:

            duplicate_results.append(
                {
                    "chunk_id":
                        _lookup(
                            result,
                            parsed_content,
                            "chunk_id",
                        ),

                    "duplicate_of_rank":
                        duplicate_rank,
                }
            )

            continue


        document_title = (
            _first_nonempty(
                _lookup(
                    result,
                    parsed_content,
                    "document_title",
                ),

                _lookup(
                    result,
                    parsed_content,
                    "title",
                ),

                _lookup(
                    result,
                    parsed_content,
                    "section_title",
                ),
            )
        )


        source_url = (
            _first_nonempty(
                _lookup(
                    result,
                    parsed_content,
                    "official_source_url",
                ),

                _lookup(
                    result,
                    parsed_content,
                    "canonical_source_url",
                ),

                _lookup(
                    result,
                    parsed_content,
                    "source_url",
                ),
            )
        )


        evidence_item = {

            "rank":
                len(
                    evidence
                )
                + 1,

            "chunk_id":
                _lookup(
                    result,
                    parsed_content,
                    "chunk_id",
                ),

            "document_id":
                _lookup(
                    result,
                    parsed_content,
                    "document_id",
                ),

            "document_title":
                document_title,

            "section_id":
                _lookup(
                    result,
                    parsed_content,
                    "section_id",
                ),

            "section_title":
                _lookup(
                    result,
                    parsed_content,
                    "section_title",
                ),

            "section_heading":
                _lookup(
                    result,
                    parsed_content,
                    "section_heading",
                ),

            "source_authority":
                _lookup(
                    result,
                    parsed_content,
                    "source_authority",
                ),

            "issuing_agency":
                _lookup(
                    result,
                    parsed_content,
                    "issuing_agency",
                ),

            "jurisdiction":
                _lookup(
                    result,
                    parsed_content,
                    "jurisdiction",
                ),

            "state":
                _lookup(
                    result,
                    parsed_content,
                    "state",
                ),

            "discipline":
                _lookup(
                    result,
                    parsed_content,
                    "discipline",
                ),

            "provider_type":
                _lookup(
                    result,
                    parsed_content,
                    "provider_type",
                ),

            "version":
                _lookup(
                    result,
                    parsed_content,
                    "version",
                ),

            "regulatory_status":
                _lookup(
                    result,
                    parsed_content,
                    "regulatory_status",
                ),

            "effective_date":
                _lookup(
                    result,
                    parsed_content,
                    "effective_date",
                ),

            "source_url":
                source_url,

            "passage":
                passage,

            "semantic_score":
                result.get(
                    "@search.rerankerScore"
                ),
        }


        evidence.append(
            evidence_item
        )


        if len(
            evidence
        ) >= top:

            break


    # ========================================================
    # SEMANTIC ANSWERS
    # ========================================================

    semantic_answers = []

    for answer in raw_body.get(
        "@search.answers",
        [],
    ):

        if not isinstance(
            answer,
            dict,
        ):
            continue

        text = answer.get(
            "text"
        )

        if not text:
            continue

        if _looks_like_serialized_json(
            text
        ):
            continue

        cleaned_text = (
            _clean_text(
                text,
                max_length=1500,
            )
        )

        if not cleaned_text:
            continue

        semantic_answers.append(
            {
                "text":
                    cleaned_text,

                "score":
                    answer.get(
                        "score"
                    ),
            }
        )


    # ========================================================
    # RESPONSE CONTRACT
    # ========================================================

    return {

        "status":
            "success",

        "tool":
            "search_regulatory_content",

        "read_only":
            True,

        "query":
            query,

        "retrieval": {

            "provider":
                "Azure AI Search",

            "endpoint":
                SEARCH_ENDPOINT,

            "index":
                INDEX_NAME,

            "query_type":
                "semantic",

            "semantic_configuration":
                SEMANTIC_CONFIGURATION,

            "requested_results":
                top,

            "raw_results_examined":
                len(
                    raw_results
                ),

            "excluded_results":
                len(
                    excluded_results
                ),

            "duplicates_removed":
                len(
                    duplicate_results
                ),

            "empty_passages_removed":
                len(
                    empty_passage_results
                ),

            "json_passages_removed":
                len(
                    json_passage_results
                ),

            "evidence_returned":
                len(
                    evidence
                ),
        },

        "semantic_answers":
            semantic_answers,

        "result_count":
            len(
                evidence
            ),

        "results":
            evidence,
    }


# ============================================================
# MCP TOOL
# ============================================================

@mcp.tool
def search_regulatory_content(
    query: str,
    top: int = 5,
) -> dict[str, Any]:
    """
    Search authoritative regulatory content using TMC's
    read-only Azure AI Search regulatory index.

    Use this tool for CMS, Medicare, skilled therapy, SNF/NF,
    medical necessity, documentation, medical review, TPE,
    ADR, and related regulatory questions.

    Args:
        query:
            Natural-language regulatory question.

        top:
            Number of distinct evidence passages to return.
            Minimum 1, maximum 10.

    Returns:
        Clean, deduplicated regulatory evidence including
        document title, section, authority, agency, source URL,
        evidence passage, and semantic relevance score.

    This tool is read-only.
    """

    return search_regulatory_content_impl(
        query=query,
        top=top,
    )


# ============================================================
# LOCAL TEST
# ============================================================

def run_local_test(
    query: str,
    top: int,
) -> None:

    print()
    print("=" * 72)
    print("TMC REGULATORY SEARCH MCP - PACKAGE TEST")
    print("=" * 72)


    if not os.getenv(
        "AZURE_CLIENT_SECRET"
    ):

        secret = getpass(
            "Enter regulatory search runtime client secret: "
        )

        os.environ[
            "AZURE_CLIENT_SECRET"
        ] = secret


    print()
    print(
        "Tool:",
        "search_regulatory_content",
    )

    print(
        "Query:",
        query,
    )

    print()


    try:

        result = (
            search_regulatory_content_impl(
                query=query,
                top=top,
            )
        )

    except Exception as exc:

        print(
            "TEST FAILED:"
        )

        print(
            type(exc).__name__,
            str(exc),
        )

        raise SystemExit(1)


    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


    results = result.get(
        "results",
        [],
    )

    clean_passages = all(

        item.get(
            "passage"
        )

        and

        not _looks_like_serialized_json(
            item.get(
                "passage"
            )
        )

        for item
        in results
    )


    overall = (

        result.get(
            "tool"
        )
        == "search_regulatory_content"

        and

        result.get(
            "read_only"
        )
        is True

        and

        len(
            results
        ) > 0

        and

        clean_passages
    )


    print()
    print("=" * 72)
    print("PACKAGE TEST SUMMARY")
    print("=" * 72)

    print(
        "Tool contract valid:",
        result.get(
            "tool"
        )
        == "search_regulatory_content",
    )

    print(
        "Read-only contract valid:",
        result.get(
            "read_only"
        )
        is True,
    )

    print(
        "Evidence passages:",
        len(
            results
        ),
    )

    print(
        "Clean prose passages:",
        clean_passages,
    )

    print()
    print(
        "OVERALL PACKAGE TEST PASSED:",
        overall,
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "TMC Regulatory Search MCP Server"
        )
    )


    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run regulatory search directly instead "
            "of starting the MCP STDIO server."
        ),
    )


    parser.add_argument(
        "--query",
        default=(
            "What are the Medicare requirements "
            "for documenting medical necessity "
            "for skilled therapy?"
        ),
    )


    parser.add_argument(
        "--top",
        type=int,
        default=5,
    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # Local package test
    # --------------------------------------------------------

    if args.test:

        run_local_test(
            query=args.query,
            top=args.top,
        )

        return


    # --------------------------------------------------------
    # TrueFoundry Hosted STDIO
    # --------------------------------------------------------

    if not os.getenv(
        "AZURE_CLIENT_SECRET"
    ):

        raise RuntimeError(
            "AZURE_CLIENT_SECRET must be configured "
            "before starting the MCP server."
        )


    # FastMCP defaults to STDIO here.
    mcp.run()


if __name__ == "__main__":
    main()