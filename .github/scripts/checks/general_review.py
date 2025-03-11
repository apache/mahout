import ast
from github import Github

class GeneralReviewCheck:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def process_pr_file(self, file, repo, pr):
        comments = []
        analyzed = False

        if file.filename.endswith('.py'):
            analyzed = True
            content = repo.get_contents(file.filename, ref=pr.head.sha).decoded_content.decode()

            # Basic code analysis
            try:
                tree = ast.parse(content)
                analysis = self.analyze_code_structure(tree)
                if analysis:
                    comments.append({
                        "path": file.filename,
                        "body": analysis,
                        "line": 1
                    })
            except SyntaxError as e:
                comments.append({
                    "path": file.filename,
                    "body": f"⚠️ Syntax error found:\n{e}",
                    "line": 1
                })

        return comments, analyzed

    def analyze_code_structure(self, tree):
        analysis = []

        # Check for large classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 10:
                    analysis.append(f"- Class '{node.name}' has {len(methods)} methods. Consider splitting into smaller classes.")

                    # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if lines > 50:
                    analysis.append(f"- Function '{node.name}' is {lines} lines long. Consider breaking it into smaller functions.")

                    # Check for TODO comments
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and 'TODO' in node.value.value.upper():
                    analysis.append(f"- TODO comment found: {node.value.value}")

        if analysis:
            return "🔍 Code structure analysis:\n" + "\n".join(analysis)
        return None

    def generate_general_impression(self, pr):
        prompt = f"""Provide a general impression of this pull request. Consider:  
- Overall code quality  
- Code organization  
- Documentation  
- Potential improvements  
- Areas that need special attention  
  
Pull Request Details:  
Title: {pr.title}  
Description: {pr.body}  
Files Changed: {pr.changed_files}  
Additions: {pr.additions}  
Deletions: {pr.deletions}  
  
Provide your analysis in markdown format with these sections:  
1. Overall Impression  
2. Strengths  
3. Areas for Improvement  
4. Special Attention Needed"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "system",
                "content": "You are an experienced code reviewer. Provide a high-level analysis of pull requests."
            }, {
                "role": "user",
                "content": prompt
            }]
        )

        return response.choices[0].message.content

    def process_pr(self, pr):
        general_analysis = self.generate_general_impression(pr)
        return [{
            "path": "GENERAL",
            "body": general_analysis,
            "line": 0
        }], True  