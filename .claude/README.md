# GitHub Workflow Orchestrator - Nested Agent Architecture

This directory contains a specialized nested agent architecture for comprehensive codebase improvements and GitHub workflow automation.

## Architecture Overview

The system consists of **1 main orchestrator** and **8 specialized agents** that work together to deliver comprehensive improvements:

```
github-workflow-orchestrator (main coordinator)
│
├── Core Development Agents
│   ├── codebase-analyzer          - Deep analysis of code structure
│   ├── feature-implementer        - Code implementation
│   └── test-runner                - Test execution and verification
│
├── Code Quality Agents
│   ├── lint-fixer                 - Fix linting issues
│   ├── formatter                  - Code formatting
│   └── knowledge-extractor        - Documentation and ADRs
│
└── GitHub & Git Agents
    ├── github-issue-manager       - Issue creation and management
    └── git-workflow-manager       - Git operations and PRs
```

## Agent Specifications

### Main Orchestrator

**File:** [github-workflow-orchestrator.md](github-workflow-orchestrator.md)

**Role:** Coordinates all other agents, manages workflow execution, handles parallel processing

**When to use:** User requests comprehensive improvements, bug fixes, feature implementation, or GitHub workflow automation

**Key Responsibilities:**
- Analyze user requests
- Determine which agents to invoke
- Orchestrate parallel/sequential execution
- Coordinate between agents
- Synthesize comprehensive summaries

### Core Development Agents

#### codebase-analyzer

**File:** [codebase-analyzer.md](codebase-analyzer.md)

**Role:** Deep code analysis, architecture understanding, issue detection

**Tools:** Read, Grep, Glob, Bash

**Key Tasks:**
- Map codebase structure
- Identify architectural patterns
- Detect performance issues
- Find security vulnerabilities
- Analyze bundle sizes
- Check test coverage

**Output:** Structured analysis report with prioritized issues and recommendations

#### feature-implementer

**File:** [feature-implementer.md](feature-implementer.md)

**Role:** Write clean, maintainable code for features and bug fixes

**Tools:** Read, Edit, Write, Grep, Glob, Bash

**Key Tasks:**
- Implement new features
- Fix bugs
- Refactor code
- Follow existing patterns
- Maintain type safety
- Handle errors gracefully

**Output:** Implementation summary with files modified and test results

#### test-runner

**File:** [test-runner.md](test-runner.md)

**Role:** Execute tests, analyze failures, verify fixes

**Tools:** Bash, Read, Grep

**Key Tasks:**
- Run test suites
- Analyze failures
- Check coverage
- Diagnose root causes
- Provide fix recommendations

**Output:** Test results report with pass/fail status and coverage metrics

### Code Quality Agents

#### lint-fixer

**File:** [lint-fixer.md](lint-fixer.md)

**Role:** Fix linting issues and enforce code quality standards

**Tools:** Bash, Read, Edit, Write

**Key Tasks:**
- Run ESLint
- Auto-fix issues
- Manually fix complex issues
- Handle unused variables
- Improve type safety
- Educate on best practices

**Output:** Linting status report with before/after comparison

#### formatter

**File:** [formatter.md](formatter.md)

**Role:** Ensure consistent code formatting with Prettier

**Tools:** Bash, Read, Edit, Write

**Key Tasks:**
- Run Prettier
- Format all files
- Configure formatting rules
- Handle file-specific overrides
- Integrate with editors

**Output:** Formatting report with files processed and verification status

#### knowledge-extractor

**File:** [knowledge-extractor.md](knowledge-extractor.md)

**Role:** Extract and document architecture, patterns, and decisions

**Tools:** Read, Grep, Glob, Write, Bash

**Key Tasks:**
- Create architecture documentation
- Write ADRs (Architecture Decision Records)
- Document components and APIs
- Create onboarding guides
- Extract design patterns

**Output:** Comprehensive documentation files (MD, ADR templates)

### GitHub & Git Agents

#### github-issue-manager

**File:** [github-issue-manager.md](github-issue-manager.md)

**Role:** Create, update, and manage GitHub issues with proper templates

**Tools:** Bash, Read, Write

**Key Tasks:**
- Create issues with templates
- Add labels and metadata
- Update issue status
- Close resolved issues
- Link related issues

**Output:** GitHub issues with proper structure and tracking

#### git-workflow-manager

**File:** [git-workflow-manager.md](git-workflow-manager.md)

**Role:** Manage Git operations, branches, commits, and PRs

**Tools:** Bash, Read

**Key Tasks:**
- Create and manage branches
- Write conventional commits
- Create pull requests
- Handle merges
- Clean up branches

**Output:** Git operations summary with branch/commit/PR details

## Workflow Patterns

### Standard Improvement Workflow

```
1. codebase-analyzer
   ↓
2. knowledge-extractor (parallel with #1)
   ↓
3. github-issue-manager (depends on #1-2)
   ↓
4. feature-implementer
   ↓
5. lint-fixer (parallel with #4-6)
   ↓
6. formatter (parallel with #4-6)
   ↓
7. test-runner
   ↓
8. git-workflow-manager
```

### Bug Fix Workflow

```
1. codebase-analyzer (understand bug)
   ↓
2. test-runner (reproduce and document)
   ↓
3. feature-implementer (fix bug)
   ↓
4. test-runner (verify fix)
   ↓
5. git-workflow-manager (commit)
```

## Configuration

### Agent Permissions

| Agent | Permission Mode | Rationale |
|-------|----------------|-----------|
| github-workflow-orchestrator | `default` | Coordinating role, needs user interaction |
| codebase-analyzer | `dontAsk` | Read-only analysis, no changes |
| feature-implementer | `acceptEdits` | Makes code changes |
| lint-fixer | `acceptEdits` | Makes code changes |
| formatter | `acceptEdits` | Makes formatting changes |
| test-runner | `dontAsk` | Read-only verification |
| knowledge-extractor | `dontAsk` | Read-only documentation |
| github-issue-manager | `acceptEdits` | Creates GitHub issues |
| git-workflow-manager | `acceptEdits` | Manages Git operations |

### Agent Models

| Agent | Model | Rationale |
|-------|-------|-----------|
| github-workflow-orchestrator | `sonnet` | Coordination requires reasoning |
| codebase-analyzer | `sonnet` | Deep analysis needs intelligence |
| feature-implementer | `sonnet` | Code generation needs understanding |
| Most others | `sonnet` | General-purpose capability |
| Simple tasks | `haiku` | Can be specified per invocation |

## Usage Examples

### Invoke Orchestrator for Comprehensive Analysis

```
User: "Perform extensive codebase improvements"

Orchestrator:
1. Invokes codebase-analyzer
2. Invokes knowledge-extractor (parallel)
3. Creates issues via github-issue-manager
4. Implements fixes via feature-implementer
5. Runs tests via test-runner
6. Formats code via formatter
7. Commits via git-workflow-manager
8. Provides comprehensive summary
```

### Invoke Individual Agent

```
User: "Fix all linting issues"

Directly invoke lint-fixer agent
```

## Benefits of Nested Architecture

### 1. Specialization
Each agent focuses on one domain, leading to:
- Deeper expertise in each area
- More targeted tools and permissions
- Better context management
- Specialized prompt engineering

### 2. Parallel Processing
Independent agents can run simultaneously:
- codebase-analyzer + knowledge-extractor
- lint-fixer + formatter + feature-implementer
- Multiple test runners on different suites

### 3. Modularity
Easy to:
- Add new agents without modifying existing ones
- Update individual agents independently
- Test agents in isolation
- Reuse agents in different orchestrators

### 4. Error Isolation
Failures are contained:
- One agent failing doesn't stop others
- Easy to identify which agent has issues
- Can retry individual agents
- Partial progress is preserved

### 5. Token Efficiency
Each agent carries only relevant context:
- No need to pass formatting rules to code analyzer
- Test runner doesn't need Git history
- Smaller contexts = faster processing

## Best Practices

### For Orchestrator
1. Always use TodoWrite to track progress
2. Run agents in parallel when possible
3. Provide clear progress updates to user
4. Pass context between agents effectively
5. Synthesize comprehensive summaries

### For Individual Agents
1. Stay focused on single responsibility
2. Use appropriate tools and permissions
3. Provide structured output
4. Include file references with line numbers
5. Suggest next steps

### For Agent Communication
1. Include relevant context from previous agents
2. Specify output format clearly
3. Track dependencies between agents
4. Handle agent failures gracefully
5. Maintain clear audit trail

## Extending the Architecture

### Adding New Agents

1. Create new `.md` file in `.claude/agents/`
2. Follow agent template structure:
   ```yaml
   ---
   name: agent-name
   description: When to use this agent
   tools: tool1, tool2, tool3
   model: sonnet
   permissionMode: default
   ---

   Agent system prompt...
   ```

3. Update orchestrator to include new agent
4. Update this README with agent specification

### Agent Template

```markdown
---
name: my-specialist-agent
description: Brief description of when this agent should be invoked
tools: Read, Write, Bash
model: sonnet
permissionMode: default
---

You are a [domain] specialist responsible for [main responsibility].

## Your Mission
[What problems does this agent solve?]

## Workflow
[Step-by-step process]

## Output Format
[Expected output structure]

## Best Practices
[Key guidelines]
```

## Maintenance

### Regular Updates
- Review agent performance quarterly
- Update prompts based on usage patterns
- Add new tools as needed
- Refine orchestration strategies

### Monitoring
Track agent effectiveness:
- Success rate of tasks
- Time to completion
- User satisfaction
- Error frequency

### Documentation
Keep this README updated:
- Add new agents
- Update workflow patterns
- Document lessons learned
- Share best practices

## Related Resources

- [Subagents Documentation](https://code.claude.com/docs/en/sub-agents)
- [Agent SDK Documentation](https://docs.anthropic.com/claude-agent-sdk)
- [Claude Code Best Practices](https://code.claude.com/docs/en/best-practices)

## License

These agent definitions are part of the project and follow the same license.
