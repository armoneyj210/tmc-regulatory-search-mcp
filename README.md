# TMC Regulatory Search MCP

Read-only MCP server for the TMC CMS Regulatory Assistant POC.

## Purpose

The server exposes one MCP tool:

`search_regulatory_content`

The tool performs semantic retrieval against the TMC Azure AI Search regulatory index.

## Architecture

TrueFoundry Agent
→ TrueFoundry MCP Gateway
→ Hosted STDIO MCP
→ search_regulatory_content
→ Microsoft Entra ID
→ Azure AI Search
→ regulatory-rag-chunks-index

## Required environment variables

- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX`
- `AZURE_SEARCH_API_VERSION`
- `AZURE_SEARCH_SEMANTIC_CONFIGURATION`

Only `AZURE_CLIENT_SECRET` is required when using the repository defaults for the POC.

## MCP tool

### search_regulatory_content

Arguments:

- `query`: natural-language regulatory question
- `top`: number of distinct evidence passages, from 1 through 10

The tool is read-only.

It returns:

- document title
- section
- issuing agency
- source authority
- effective date
- source URL
- evidence passage
- semantic score
- retrieval observability metadata

## Local test

After installation:

```powershell
tmc-regulatory-search --test