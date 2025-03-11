#!/usr/bin/env python3
import os
import sys
from github import Github
from gofannon.github.pr_review_tool import PRReviewTool

def check_env_vars():
    required_vars = [
        'GITHUB_TOKEN', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL_NAME', 'PR_NUMBER', 'REPO_NAME'
    ]
    for var in required_vars:
        if not os.environ.get(var):
            sys.exit(f"Error: Required environment variable '{var}' is missing.")

check_env_vars()

def main():
    pr_number = int(os.environ['PR_NUMBER'])
    repo_name = os.environ['REPO_NAME']
    pr_review_tool = PRReviewTool()
    review_summary = pr_review_tool.fn(pr_number=pr_number, repo_name=repo_name)

    # Post the review comment to the pull request using PyGithub.
    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(review_summary)

if __name__ == "__main__":
    main()
