import json
import requests
import schedule
import time
import threading

# Documentation de l'API de Github: https://docs.github.com/en/rest?apiVersion=2022-11-28

GITHUB_PAT = "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"  # Genérez un token dans github et collez le ici
GITHUB_URL = "https://api.github.com/repos"
REPO = "hive"
OWNER = "apache"
HEADERS = {
	"Accept": "application/vnd.github+json",
	"Authorization": "token " + GITHUB_PAT
}


def extract_commits(page: int = 1, per_page: int = 100) -> list[dict]:
    """Extract a list of commits.

    Args:
        page (int, optional): The page number of the results to fetch. Defaults to 1.
        per_page (int, optional): The number of results per page (max 100). Defaults to 100.

    Returns:
        list[dict]: List of dictionnaries containning:
            - "message": The message of the commit.
            - "sha": Commit hash. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits?per_page={per_page}&page={page}", headers=HEADERS)
        response.raise_for_status()
        return [{"sha": e["sha"], "message": e["commit"]["message"]} for e in json.loads(response.text)]
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"An error occurred while fetching commits:\n\n\t\t{e}\n")
        return []


def extract_commit_content(hash: str) -> list[str]:
    """Extract the content of a single commit.

    Args:
        hash (str): Commit reference.

    Returns:
        list[str]: List of modified Java and C++ filenames. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits/" + hash, headers=HEADERS)
        commit = json.loads(response.text)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"An error occurred while fetching commit {hash}:\n\n\t\t{e}\n")
        return []
    return [e["filename"] for e in commit["files"] \
            if (e["filename"].endswith(".java") or e["filename"].endswith(".cpp")) and e["status"] == "modified"]


def extract_tags(page: int = 1, per_page: int = 100) -> list[dict]:
    """Extract GitHub tags.

    Args:
        page (int, optional): The page number of the results to fetch. Defaults to 1.
        per_page (int, optional): The number of results per page (max 100). Defaults to 100.

    Returns:
        list[dict]: List of dictionnaries containning:
            - tagName: Tag name.
            - commitId: Commit hash.
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/tags?per_page={per_page}&page={page}", headers=HEADERS)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"An error occurred while fetching tags:\n\n\t\t{e}\n")
        return []
    return [
        {
            "tagName": e["name"],
            "commitId": e["commit"]["sha"]
        }
        for e in json.loads(response.text)
    ]
