# Machine registry and health service

`AssistantStorage` stores canonical machine records without inventing operational machines. `assistant.health.machine_health` separates model-estimated sensor evidence, confirmed/pending technician observations, and unknown information. FD001 RUL is not converted into a health percentage or diagnosis. A missing registry record returns an explicit unavailable response.
