from collections import defaultdict
from pathlib import Path
import threading
import time
import schedule

from modules.jira_extractor import extract_issues
from modules.commit_extractor import extract_commit_content, extract_commits, extract_tags

JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = FIXED AND affectedVersion ~ \"2.*\""
OUTPUT_FILE = Path("commits.json")  # Extraire les commits dans ce fichier

def _extract_commits(issues: list[str]) -> list[dict]:
	"""Extract commits by issues.

	Args:
		issues (list[str]): List of JIRA issue keys.

    Returns:
        list[dict]: List of dictionnaries containning:
            - key (str): Jira key.
            - commit (str): Commit ID.
			- modifiedFiles (list[str]): List of modified filenames.
	"""
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
	commits = extract_commits(page=page, per_page=per_page)
	while commits or len(found) != len(issues):
		for commit in commits:
			message = commit["message"]
			for ticket in issues:
				if ticket in message:
					commit["files"] = extract_commit_content(commit["sha"])
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
		commits = extract_commits(page=page, per_page=100)
	stop_event.set()
	scheduler_thread.join()
	return [
		{
			"key": issue_key,
			"commit": commit["sha"],
			"modifiedFiles": commit["files"]
		}
		for issue_key, commit in found.items()
	]


print("Extracting jira issues...")
issues = {}
for issue in extract_issues(JIRA_SEARCH_FILTER, ["versions"]):
	issues[issue["key"]] = {"affectedVersions": [e["name"] for e in issue["fields"]["versions"]]}
print(f"\tFound {len(issues)} issues.\n")

print("Extracting github commits...")
commits = _extract_commits(issues.keys())
print(f"\tFound {len(commits)} commits.\n")

files_with_bugs = defaultdict(set)
for commit in commits:
    files = commit.get("modifiedFiles", [])
    if not files:
        continue
    affected_versions = issues[commit["key"]].get("affectedVersions", [])
    for version in affected_versions:
        files_with_bugs[version].update(files)

del commits
del issues

# import re
# pattern = re.compile("(rel/)?release-2\\.[0-9]\\.0$")
# tags = [e for e in extract_tags() if re.match(pattern, (e["tagName"]))]
# print(tags)
