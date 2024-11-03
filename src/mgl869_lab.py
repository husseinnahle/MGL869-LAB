import json
from pathlib import Path
from modules.jira_extractor import extract_issues
from modules.commit_extractor import extract_commits

JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = FIXED AND affectedVersion = 2.0.0"
OUTPUT_FILE = Path("commits.json")  # Extraire les commits dans ce fichier

print("Extracting jira issues...")
issues = [issue["key"] for issue in extract_issues(JIRA_SEARCH_FILTER, ["versions"])]
print(f"\tFound {len(issues)} issues.\n")

print("Extracting github commits...")
commits = extract_commits(issues)
print(f"\tFound {len(commits)} commits.\n")

files_with_bugs = set()
for e in commits:
	files_with_bugs.update(e["files"])

with open(OUTPUT_FILE, "w") as file:
	json.dump(list(files_with_bugs), file, indent=4)
