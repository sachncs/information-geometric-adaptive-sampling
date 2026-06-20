# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please report it
responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

Please send a description of the vulnerability to **[INSERT EMAIL]**. You should
receive a response within 48 hours. If for some reason you do not, please follow
up to ensure we received your original message.

Please include the following information in your report:

- Type of vulnerability (e.g., buffer overflow, code injection, etc.)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Disclosure Policy

When we receive a security report, we will:

1. Confirm the vulnerability and determine its impact.
2. Audit related code for any similar issues.
3. Prepare a fix and release it as soon as possible.
4. Publish a security advisory on GitHub once the fix is released.

We will coordinate with the reporter on the disclosure timeline and will credit
reporters who follow responsible disclosure practices.

## Security Best Practices

When using this library in production:

- Keep dependencies up to date.
- Validate all input data before passing it to the sampler.
- Be aware that this package uses floating-point arithmetic and may be subject
  to numerical stability issues with extreme parameter values.
- Do not use this package for safety-critical applications without thorough
  testing and validation.

## Scope

This security policy applies to the code distributed through this repository's
official releases. It does not apply to:

- Third-party forks or derivatives
- Code in examples/ directory (educational purposes only)
- The simplified model approximations (not intended for production use)
