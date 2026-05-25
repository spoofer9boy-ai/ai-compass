# Git Version Control

**Phase:** PHASE-01-foundations  
**Prerequisites:** []  
**Estimated Time:** 40 minutes

## Why am I learning this?

You will spend more time reading, reviewing, and reverting code than writing it from scratch. In a team of AI engineers, three people might touch the same training script in a single day. Without version control, "it works on my machine" becomes a permanent state of affairs, and experiments become impossible to reproduce because nobody remembers which notebook produced which model weights.

Git is the de facto standard for tracking changes in software. It is not a backup tool, though it can serve as one. It is a directed graph of snapshots that lets you move through time, branch into parallel realities, and merge them back together. Every open-source model, every PyTorch release, and every production API you will ever depend on lives in a Git repository. Learning Git means learning to work with the rest of the industry on its own terms.

The honest truth: Git's user interface is famously inconsistent. The same flag means different things on different commands, and the documentation often assumes you already understand the underlying object model. But the core concepts—commits, branches, merges—are simple, robust, and universal. This file focuses on the 20% of Git that covers 80% of your daily work.

## Where will I be using it?

- **Model Development:** Tracking changes to training code, hyperparameter configs, and data preprocessing pipelines so experiments are reproducible.
- **Open-Source Contribution:** Submitting pull requests to HuggingFace Transformers, PyTorch, or scikit-learn.
- **Collaborative Research:** Sharing notebooks and scripts with colleagues without emailing ZIP files.
- **Production Deployment:** Tagging releases, rolling back broken deployments, and maintaining hotfix branches.
- **Experiment Tracking:** Pairing Git commits with MLflow or Weights & Biases runs to know exactly which code produced which metric.

## Resources

- [Pro Git Book](https://git-scm.com/book/en/v2) — The official, free, comprehensive guide to Git. Chapters 1–3 cover everything in this file.
- [GitHub Docs: Git and GitHub Learning Resources](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources) — Curated path from GitHub for getting started.
- [GitHub Blog: How We Use GitHub](https://github.blog/engineering/engineering-principles/how-we-use-github-to-be-more-productive-collaborative-and-secure/) — How GitHub's own engineering team uses GitHub for daily workflows.
- [Git Reference Manual](https://git-scm.com/docs) — Authoritative documentation for every Git command.

## Appendix

### Core Concepts

- **Repository (repo):** A directory that Git tracks, containing your project files and the `.git` folder with the full history.
- **Commit:** A snapshot of your files at a specific point in time, identified by a SHA-1 hash. Think of it as a save point you can return to.
- **Branch:** A lightweight movable pointer to a commit. The default branch is usually called `main` or `master`.
- **Staging Area (Index):** A intermediate zone where you prepare changes before committing them. `git add` puts files here; `git commit` saves them.
- **Remote:** A version of your repository hosted on the internet or another network, such as GitHub, GitLab, or an internal server.

### The Daily Workflow

```bash
# 1. Check what changed
$ git status

# 2. Stage specific files
$ git add src/train.py config.yaml

# 3. Commit with a descriptive message
$ git commit -m "feat: add learning rate warmup to training loop"

# 4. Push to the remote repository
$ git push origin main
```

### Common Pitfalls

- **Committing large files:** Git is not designed for datasets or model checkpoints. Use `.gitignore` to exclude `.pt`, `.pkl`, and `data/` directories. For large files, use Git LFS or DVC.
- **Committing secrets:** Never commit API keys, database passwords, or cloud credentials. Once pushed, they are in the history forever. Use environment variables or secret managers.
- **Panic reverting:** `git reset --hard` permanently discards uncommitted changes. If you are unsure, use `git stash` instead.
- **Merge conflict avoidance:** Pulling from `main` before starting work and committing frequently reduces the size and pain of merge conflicts.

### Useful Commands

| Command | Purpose |
|---------|---------|
| `git log --oneline --graph` | Visualize branch history compactly. |
| `git diff` | See unstaged changes. |
| `git diff --staged` | See changes already staged for commit. |
| `git checkout -b feature-name` | Create and switch to a new branch. |
| `git pull --rebase origin main` | Update your branch with the latest remote changes cleanly. |
| `git clone <url>` | Copy a remote repository to your local machine. |

### Further Reading

- [Oh Shit, Git!?!](https://ohshitgit.com/) — Practical recovery commands for common mistakes.
- [GitHub Skills](https://skills.github.com/) — Interactive, hands-on GitHub courses.
