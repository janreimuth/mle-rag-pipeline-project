# API Rate Limits

Northstar's project API accepts 60 requests per minute per API key. A response
with HTTP status `429` indicates that the limit has been reached.

Clients receiving `429` should wait and retry using exponential backoff.
Repeated immediate requests increase load and may extend the disruption. API
keys must not be pasted into tickets, chat messages, or generation prompts.
