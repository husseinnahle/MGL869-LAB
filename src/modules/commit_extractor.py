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


def _extract_commits(page: int = 1, per_page: int = 100) -> list[dict]:
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


def _extract_commit_content(hash: str) -> list[str]:
    """Extract the content of a single commit.

    Args:
        hash (str): Commit reference.

    Returns:
        list[str]: List of modified Java and C++ filenames. 
    """
    try:
        response = requests.get(f"{GITHUB_URL}/{OWNER}/{REPO}/commits/" + hash, headers=HEADERS)
        commit = json.loads(response.text)
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return []
    return [e["filename"] for e in commit["files"] \
            if (e["filename"].endswith(".java") or e["filename"].endswith(".cpp")) and e["status"] == "modified"]


def extract_commits(issues: list[str]) -> list[dict]:

    def _print_commit_count():
        print(f"\tFound {round(len(found)*100/len(issues), 2)}% commits.")

    def _start_scheduler():
        schedule.every(10).seconds.do(_print_commit_count)
        while not stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    stop_event = threading.Event()
    scheduler_thread = threading.Thread(target=_start_scheduler)
    scheduler_thread.start()
    found = {}
    max_tries = 20
    min_found = int(0.9 * len(issues))
    try_count = 0
    previous_found_count = 0
    page = 1
    per_page = 100
    commits = _extract_commits(page=page, per_page=per_page)

    while commits or len(found) != len(issues):

        for commit in commits:
            message = commit["message"]
            for ticket in issues:
                if ticket in message:
                    commit["files"] = _extract_commit_content(commit["sha"])
                    found[ticket] = commit
                    break

        if len(found) == previous_found_count and len(found) > min_found:
            try_count += 1
            if try_count >= max_tries:
                not_found = set(issues) - set(found.keys())
                print(f"\tMax tries reached... {len(issues) - len(found)} ticket(s) not found: {','.join(not_found)}")
                break

        else:
            try_count = 0
            previous_found_count = len(found)

        page += 1
        commits = _extract_commits(page=page, per_page=100)

    stop_event.set()
    scheduler_thread.join()
    return [
        {
            "key": issue_key,
            "commitId": commit["sha"],
            "files": commit["files"]
        }
        for issue_key, commit in found.items()
    ]
