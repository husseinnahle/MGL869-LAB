from pathlib import Path
import csv
import os

from modules.jira_extractor import extract_issues
from modules.commit_extractor_localDirectory import get_commits

# JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = FIXED AND affectedVersion ~ \"2.*\""
JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = Fixed AND affectedVersion = 2.0.0"
OUTPUT_FILE = Path("Bugs_2.0.0.csv")  # Extraire les commits dans ce fichier


def write_commits_to_csv(commit_list, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Delete the file if it exists
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['key', 'filename'])

        for commit in commit_list:
            key = commit['key']
            for bug_file in commit['files']:
                writer.writerow([key, bug_file])


print("Extracting jira issues...")
issues = {}
for issue in extract_issues(JIRA_SEARCH_FILTER, ["versions"]):
    issues[issue["key"]] = {"affectedVersions": [e["name"] for e in issue["fields"]["versions"]]}
print(f"\tFound {len(issues)} issues.\n")

# Note: Hive doit être sur la main branch pour le code bonne les bons résultats
print("Extracting github commits...")
repo_path = r'C:\Users\lafor\Desktop\ETS - Cours\MGL869-01_Sujets speciaux\Laboratoire\Hive\hive'
print(repo_path)
commits_list = get_commits(repo_path, issues)

write_commits_to_csv(commits_list, OUTPUT_FILE)

del commits_list
del issues
