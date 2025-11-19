# Security Guidelines for Spot Robot Credentials

## CRITICAL: Never Commit Credentials

**DO NOT** commit any files containing:
- Usernames
- Passwords
- API tokens
- SSH keys
- Any authentication credentials

## Safe Credential Management

### Option 1: Shell Environment (Recommended)

**Windows PowerShell** - Add to your user profile (outside repo):
```powershell
# Edit: $PROFILE
$env:BOSDYN_CLIENT_USERNAME = "your_username"
$env:BOSDYN_CLIENT_PASSWORD = "your_password"
```

**Linux/macOS** - Add to your shell profile:
```bash
# Edit: ~/.bashrc or ~/.zshrc
export BOSDYN_CLIENT_USERNAME=your_username
export BOSDYN_CLIENT_PASSWORD=your_password
```

### Option 2: .env File (Must be in .gitignore)

Create `.env` file in project root (already in .gitignore):
```
BOSDYN_CLIENT_USERNAME=your_username
BOSDYN_CLIENT_PASSWORD=your_password
```

**Verify it's ignored:**
```bash
git check-ignore .env
# Should output: .env
```

### Option 3: CLI Arguments

Pass credentials via command-line flags:
```bash
python friendly_spot_main.py --robot 192.168.80.3 --user <username> --password <password>
```

**Warning:** This leaves credentials in shell history. Use sparingly.

## Files That Must NEVER Be Committed

The `.gitignore` is configured to exclude:
- `.env` and `.env.*`
- Any file with `credentials` in the name
- Any file with `secrets` in the name
- `Activate.ps1` (may contain credentials)

## Verification Checklist

Before committing:
Before committing:
- No hardcoded credentials in source files
- No credentials in README or documentation
- `.env` files are in `.gitignore`
- `Activate.ps1` is in `.gitignore` (if it contains credentials)
- Run `git status` to verify no credential files are staged

## Emergency: Credentials Accidentally Committed

If credentials were committed to git history:

1. **Change credentials immediately** on the robot
2. **Do NOT just delete the file** - it's still in git history
3. Use `git filter-branch` or BFG Repo-Cleaner to remove from history
4. Force push (if remote exists)
5. Notify team members to re-clone

## SDK Authentication Flow

The Boston Dynamics SDK tries authentication in this order:
1. Existing authentication token (if still valid)
2. Environment variables (secure, recommended)
3. Interactive prompt (fallback)

Our code never exposes the environment variable names in error messages or logs.
