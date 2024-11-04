import json
import requests

# Documentation de l'API de JIRA: https://developer.atlassian.com/cloud/jira/platform/rest/v2/intro

JIRA_URL = "https://issues.apache.org/jira/rest/api/2"
HEADERS = {
  "Accept": "application/json"
}


def extract_issues(filter: str, fields: list[str] = []) -> list[dict]:
    """Extract JIRA issues.

    Args:
        filter (str): JQL filter.
        fields (list[str], optional): Field names to extract. Defaults to [].

    Returns:
        list[dict]: List of dictionnaries containning:
            - key (str): Jira key.
            - fields (dict[dict]): Jira fields. 
    """
    try:
        params = {
          "jql": filter,
          "maxResults": "5000",
          "fieldsByKeys": "true",
          "fields": ",".join(fields)
        }
        response = requests.get(f"{JIRA_URL}/search", headers=HEADERS, params=params)
        response.raise_for_status()
        return [{"key": e["key"], "fields": e["fields"]} for e in json.loads(response.text)["issues"]]
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        return []
