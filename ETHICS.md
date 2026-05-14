# Ethics & Data-Source Notes

This document records the data-access decisions made in this project and the reasoning behind them. It exists because some of those decisions sit in genuinely contested territory and deserve to be made deliberately rather than by default.

## What this project is

A research pipeline that monitors public social and web mentions of consumer brands and applies discourse-analysis methods (Habermasian validity-claim typology, sentiment, topic clustering) to study patterns in how people talk about those brands. Output is markdown reports for personal review. NFCU is one example brand among several (alongside USAA, PenFed, and others); the methodology is brand-agnostic.

## What this project is **not**

- Not a product, commercial service, or component of one
- Not a training corpus for any ML model — classifiers run inference on already-trained foundation models, no fine-tuning, no parameter updates
- Not a sentiment-tracking service for any institution
- Not a covert monitoring tool — public mentions only, no private content, no DMs, no inferred private characteristics
- Not used to re-identify, de-anonymize, or match users with off-platform identifiers

## Sources used and why

### Serper (Google search API) — primary discovery layer

Serper queries Google's index and returns structured search results. We use it to discover mentions across:

- News (via Serper's news endpoint)
- LinkedIn public posts
- Trustpilot reviews
- Better Business Bureau pages
- YouTube video discovery (comments fetched via YouTube Data API)
- Industry press (American Banker, CU Today, CU Times, Banking Dive)
- Reddit threads (see note below)
- General web (forums, blogs, smaller outlets)

This is a deliberate architectural choice: rather than build per-platform collectors with per-platform auth, terms, and approval gates, we use Google's already-public index of the open web as a unified discovery layer. This is the same activity a human researcher does when Googling a brand to understand market sentiment — automated, but not categorically different.

### Reddit — discovered via Serper

Reddit's Responsible Builder Policy (effective late 2025) requires explicit approval to access Reddit data through the Reddit API. This project does not use the Reddit API.

We do, however, ingest Reddit content that surfaces in Google search results via Serper. We've thought about this carefully and proceed for the following reasons:

1. **Reddit posts on the public web are public statements.** Users posting publicly accept that their words will be indexed by Google, embedded in news articles, and quoted in academic discourse analysis. The expectation of privacy for a public post is essentially nil.

2. **The activity being automated is legitimate manual research.** A person Googling "navy federal complaints" and reading the resulting Reddit threads is doing something nobody objects to. Doing this systematically at low volume for research purposes is a continuation of that legitimate activity, not a categorically new harm.

3. **The harms Reddit's policy targets are not present here.** The policy was designed to gate AI-training-corpus extraction, abusive bots, and high-volume scrapers degrading site performance. None of those describe a weekly run of 30-50 search queries returning snippets, with full-text fetches limited to ~50 high-priority items.

4. **No model training, no resale, no commercial use.** The data flow is: Google search results → classification via inference → aggregate reports. Reddit content is processed and the report is written; it is not retained as a corpus, not used for training, not licensed, not sold.

5. **The methodology is publishable academic work.** Discourse-analysis research on public consumer-brand mentions is a legitimate research area. Researchers in this area routinely use whatever publicly accessible data they can responsibly access.

### What we don't do

- We do not bypass Reddit's authentication (no scraping while logged in, no use of credentials)
- We do not impersonate users or use multiple accounts to evade rate limits
- We do not retain Reddit content beyond a rolling 90-day analytical window
- We do not republish Reddit content; quotations in reports are limited to short representative examples for cluster illustration
- We do not match Reddit handles to off-platform identifiers
- We do not derive sensitive characteristics about Reddit users (health, political affiliation, sexual orientation, etc.) — the classification operates on the content of the mention itself, not on inferences about the author

### Parallel application

A non-commercial Data API application is being submitted to Reddit through the proper channel. If approved, the project will switch to Reddit's official API for Reddit-sourced mentions, which is preferable on every dimension. Until the application is resolved, Serper-discovered Reddit content is the working source.

## Twitter / X

Not currently included. Twitter's Basic API tier is paid, and direct scraping is brittle and against terms. Until Twitter offers a research-friendly access path or the project finds a clean alternative, the Twitter signal is accepted as missing.

## Privacy commitments

- Public content only — no DMs, no private subreddits, no quarantined or banned communities, no content behind authentication
- No re-identification or de-anonymization attempts
- No inference of sensitive personal characteristics
- Author handles retained in local store (necessary for cluster construction) but not surfaced in reports beyond aggregate counts
- Local SQLite storage on a personal machine; no cloud sync, no third-party data sharing
- 90-day rolling retention; aggregated metrics retained beyond that, raw mentions deleted

## Commercial use

This project is non-commercial personal research.

If at some future point a commercial entity wished to deploy similar methodology operationally, that would be a separate matter requiring that entity's own approvals through whichever platforms' commercial-track applications apply. The methodology is publishable; any specific commercial deployment is a separate action by a separate party.

## Open questions

I'm holding the following questions open and revisit them as the project evolves:

- **Whether Serper-routed access to Reddit content respects the spirit of Reddit's policy even if not the letter.** I've concluded yes for the use case as scoped. If the volume or scope grew, I'd revisit.
- **Whether Habermasian classification of customer mentions is itself ethically meaningful or just academic packaging.** I think the framework genuinely surfaces patterns sentiment alone misses (the TRUTH/RIGHTNESS/SINCERITY split is empirically distinct), but this is the kind of claim that wants empirical defense, not just methodological assertion.
- **Whether the report format itself could become a vector for misuse** (e.g., if a hostile actor used the same pipeline to map vulnerabilities of a brand for harassment campaigns). The methodology is publicly described; that's a feature for legitimate research and a bug for misuse. I don't have a complete answer here.

I expect to update this document as the project develops and as I think more about the questions it raises.
