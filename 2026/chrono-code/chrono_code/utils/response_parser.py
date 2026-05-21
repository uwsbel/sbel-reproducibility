"""
Improved response parser for extracting code from LLM responses.

This module provides robust code extraction from various response formats
including Chain of Thought structured responses.
"""

import json
import re
import logging
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Enhanced parser for extracting code and structured data from LLM responses.
    """

    def extract_code(self, response: str) -> Tuple[str, str, bool]:
        """
        Extract Python code from LLM response with multiple strategies.

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (code, extraction_method, success)
        """
        # Strategy 1: Look for CODE: section (new Chain of Thought format)
        code, success = self._extract_with_code_marker(response)
        if success:
            return code, "code_marker", True

        # Strategy 2: Look for ```python code blocks
        code, success = self._extract_python_codeblock(response)
        if success:
            return code, "python_codeblock", True

        # Strategy 3: Look for generic ``` code blocks
        code, success = self._extract_generic_codeblock(response)
        if success:
            return code, "generic_codeblock", True

        # Strategy 4: Look for import statements as code start
        code, success = self._extract_from_import(response)
        if success:
            return code, "import_detection", True

        # Strategy 5: Try to extract between known markers
        code, success = self._extract_between_markers(response)
        if success:
            return code, "marker_detection", True

        # Strategy 6: Last resort - check if entire response looks like code
        if self._looks_like_code(response):
            logger.warning("Using entire response as code (last resort)")
            return response.strip(), "full_response", True

        # Failed to extract code
        error_code = self._generate_error_code(response)
        return error_code, "failed", False

    def extract_json_patch(self, response: str) -> Tuple[Optional[List[dict]], bool]:
        """
        Extract JSON Patch operations from LLM response.

        Args:
            response: Raw LLM response (expected to contain JSON array of RFC 6902 ops)

        Returns:
            Tuple of (patch_ops, success). On success, patch_ops is list of dicts.
            Each op must have "op" and "path"; add/replace must have "value".
        """
        raw_json: Optional[str] = None

        # Strategy 1: Extract from ```json block
        pattern = r'```(?:json)?\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            raw_json = max(matches, key=len).strip()

        # Strategy 2: Treat entire response as JSON
        if not raw_json or not raw_json.startswith("["):
            stripped = response.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                raw_json = stripped

        if not raw_json:
            return None, False

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON Patch: {e}")
            return None, False

        if not isinstance(data, list):
            logger.warning("JSON Patch must be a JSON array")
            return None, False

        valid_ops = ["add", "remove", "replace", "move", "copy", "test"]
        for i, op in enumerate(data):
            if not isinstance(op, dict):
                logger.warning(f"Patch op at index {i} is not a dict")
                return None, False
            if "op" not in op or "path" not in op:
                logger.warning(f"Patch op at index {i} missing 'op' or 'path'")
                return None, False
            if op["op"] not in valid_ops:
                logger.warning(f"Patch op at index {i} has invalid op: {op['op']}")
                return None, False
            if op["op"] in ("add", "replace") and "value" not in op:
                logger.warning(f"Patch op at index {i} (add/replace) missing 'value'")
                return None, False

        logger.info(f"Extracted JSON Patch with {len(data)} operations")
        return data, True

    def extract_api_plan(self, response: str) -> Optional[Dict[str, List[str]]]:
        """
        Extract API plan from Chain of Thought response.

        Args:
            response: LLM response containing API PLAN section

        Returns:
            Dictionary with planned APIs or None
        """
        api_plan = {
            "classes": [],
            "functions": [],
            "methods": [],
            "unverified": []
        }

        if "API PLAN:" not in response:
            return None

        # Extract API PLAN section
        try:
            parts = response.split("API PLAN:")
            if len(parts) < 2:
                return None

            api_section = parts[1]

            # Find where VERIFICATION section starts
            if "VERIFICATION:" in api_section:
                api_section = api_section.split("VERIFICATION:")[0]

            # Parse different categories
            lines = api_section.split('\n')
            current_category = None

            for line in lines:
                line = line.strip()

                if "Classes:" in line or "Classes to use:" in line:
                    current_category = "classes"
                elif "Functions:" in line or "Functions to use:" in line:
                    current_category = "functions"
                elif "Methods:" in line or "Methods to use:" in line:
                    current_category = "methods"
                elif line.startswith("- ") and current_category:
                    # Extract API name from bullet point
                    api_name = line[2:].strip()
                    # Remove any annotations like (NOT FOUND)
                    api_name = re.sub(r'\s*\([^)]*\)\s*$', '', api_name)
                    if api_name:
                        api_plan[current_category].append(api_name)

            return api_plan if any(api_plan.values()) else None

        except Exception as e:
            logger.warning(f"Failed to extract API plan: {e}")
            return None

    def extract_verification_results(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract verification results from Chain of Thought response.

        Args:
            response: LLM response containing VERIFICATION section

        Returns:
            Dictionary with verification results or None
        """
        if "VERIFICATION:" not in response:
            return None

        try:
            parts = response.split("VERIFICATION:")
            if len(parts) < 2:
                return None

            verification_section = parts[1]

            # Find where CODE section starts
            if "CODE:" in verification_section:
                verification_section = verification_section.split("CODE:")[0]

            # Parse verification results
            results = {
                "found": [],
                "not_found": [],
                "alternatives": {}
            }

            lines = verification_section.split('\n')
            for line in lines:
                line = line.strip()

                # Look for found APIs
                if "✓" in line or "FOUND" in line or "exists" in line.lower():
                    # Extract API name
                    api_match = re.search(r'`([^`]+)`', line)
                    if api_match:
                        results["found"].append(api_match.group(1))

                # Look for not found APIs
                elif "✗" in line or "NOT FOUND" in line or "not found" in line.lower():
                    # Extract API name
                    api_match = re.search(r'`([^`]+)`', line)
                    if api_match:
                        api_name = api_match.group(1)
                        results["not_found"].append(api_name)

                        # Look for alternatives
                        alt_match = re.search(r'use\s+`([^`]+)`', line.lower())
                        if alt_match:
                            results["alternatives"][api_name] = alt_match.group(1)

            return results if (results["found"] or results["not_found"]) else None

        except Exception as e:
            logger.warning(f"Failed to extract verification results: {e}")
            return None

    def _extract_with_code_marker(self, response: str) -> Tuple[str, bool]:
        """Extract code after CODE: marker."""
        if "CODE:" in response:
            parts = response.split("CODE:")
            if len(parts) > 1:
                code = parts[1].strip()

                # Remove any markdown code blocks
                if "```" in code:
                    # Extract content between first ``` and last ```
                    code_parts = code.split("```")
                    if len(code_parts) >= 2:
                        # Get the content between markers
                        code = code_parts[1]
                        # Remove language identifier if present
                        if code.startswith("python"):
                            code = code[6:].strip()

                # Clean up the code
                code = code.strip()

                if code and len(code) > 10:  # Ensure we have meaningful code
                    logger.info(f"Extracted code using CODE: marker ({len(code)} chars)")
                    return code, True

        return "", False

    def _extract_python_codeblock(self, response: str) -> Tuple[str, bool]:
        """Extract code from ```python blocks."""
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Take the longest code block
            code = max(matches, key=len)
            logger.info(f"Extracted code from python codeblock ({len(code)} chars)")
            return code.strip(), True

        return "", False

    def _extract_generic_codeblock(self, response: str) -> Tuple[str, bool]:
        """Extract code from generic ``` blocks."""
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Take the longest block that looks like Python
            for block in sorted(matches, key=len, reverse=True):
                if self._looks_like_code(block):
                    logger.info(f"Extracted code from generic codeblock ({len(block)} chars)")
                    return block.strip(), True

        return "", False

    def _extract_from_import(self, response: str) -> Tuple[str, bool]:
        """Extract code starting from import statements."""
        # Look for import statements
        import_patterns = [
            r'^import\s+\w+',
            r'^from\s+\w+\s+import',
            r'^import\s+pychrono',
        ]

        for pattern in import_patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                # Extract everything from the import to the end
                code = response[match.start():]

                # Try to clean it up - remove any obvious non-code at the end
                # Use only markers that indicate LLM explanation (NOT valid Python)
                # Do NOT use "\n\n#" - comments are valid Python code
                end_markers = [
                    "\n\nThis code",
                    "\n\nThe above",
                    "\n\nNote:",
                    "\n\nExplanation:",
                    "\n\nHere's",
                    "\n\nAbove",
                ]

                for marker in end_markers:
                    if marker in code:
                        code = code[:code.index(marker)]

                if len(code) > 50:  # Ensure we have substantial code
                    logger.info(f"Extracted code from import statement ({len(code)} chars)")
                    return code.strip(), True

        return "", False

    def _extract_between_markers(self, response: str) -> Tuple[str, bool]:
        """Try to extract code between known start/end markers."""
        # Common patterns for code sections
        start_markers = [
            "Here's the code:",
            "Here is the code:",
            "Generated code:",
            "Complete code:",
            "Fixed code:",
            "Corrected code:",
            "```",
        ]

        end_markers = [
            "This code",
            "The above",
            "Note:",
            "Explanation:",
            "---",
            "```",
        ]

        for start in start_markers:
            if start in response:
                start_idx = response.index(start) + len(start)
                code = response[start_idx:].strip()

                # Look for end marker
                for end in end_markers:
                    if end in code:
                        code = code[:code.index(end)].strip()

                if self._looks_like_code(code) and len(code) > 50:
                    logger.info(f"Extracted code between markers ({len(code)} chars)")
                    return code, True

        return "", False

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like Python code."""
        if not text or len(text) < 20:
            return False

        # Check for Python keywords
        python_keywords = [
            'import', 'from', 'def', 'class', 'if', 'for', 'while',
            'return', 'print', 'chrono', 'Ch', '=', '(', ')'
        ]

        keyword_count = sum(1 for kw in python_keywords if kw in text)

        # Check for common Python patterns
        has_imports = 'import' in text or 'from' in text
        has_functions = 'def ' in text
        has_assignments = '=' in text
        has_indentation = '\n    ' in text or '\n\t' in text

        # Score based on Python characteristics
        score = keyword_count
        if has_imports:
            score += 3
        if has_functions:
            score += 2
        if has_assignments:
            score += 1
        if has_indentation:
            score += 2

        return score >= 4

    def _generate_error_code(self, response: str) -> str:
        """Generate error code when extraction fails."""
        error_code = f'''"""
ERROR: Failed to extract code from LLM response.

The response parser tried multiple extraction strategies but could not
identify valid Python code in the response.

Response length: {len(response)} characters
Response preview: {response[:200]}...

This is a placeholder to prevent empty file generation.
Please check the dialog logs for the full LLM response.
"""

# Placeholder code to prevent compilation errors
import pychrono.core as chrono

print("ERROR: Code extraction failed. See dialog logs for details.")
'''
        return error_code