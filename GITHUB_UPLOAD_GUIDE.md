# GitHub Upload and Zenodo Archival Guide

**Purpose**: Upload DHG-LGB repository to GitHub and create permanent DOI via Zenodo
**Date**: 2025-12-07

---

## 📋 Prerequisites

- [ ] GitHub account (create at https://github.com/signup if needed)
- [ ] Git installed on your computer
- [ ] Zenodo account (sign up at https://zenodo.org using GitHub account)

---

## 🚀 Step-by-Step Instructions

### Part 1: Create GitHub Repository

#### 1.1 Create New Repository on GitHub

1. **Go to GitHub**: Navigate to https://github.com
2. **Click "New"**: In the top-right corner, click the "+" icon → "New repository"
3. **Repository Settings**:
   - **Repository name**: `DHG-LGB-Repository`
   - **Description**: `Disease-Hypergraph Integrated with LightGBM for Predicting Metabolite-Disease Associations`
   - **Visibility**: ✅ Public (required for Zenodo archival)
   - **Initialize repository**: ❌ Do NOT check "Add a README" (we already have one)
4. **Click "Create repository"**

#### 1.2 Initialize Local Repository

Open Git Bash or Command Prompt in the repository folder:

```bash
# Navigate to repository folder
cd "D:\xavier\第二次论文\DHG-LGB-Repository"

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Complete DHG-LGB framework implementation

- HGNN model with 2-layer architecture
- LightGBM classifier with 5-fold CV
- All 7 evaluation metrics (MCC, AUC, AUPRC, etc.)
- Similarity computation (Tanimoto, GO semantic, BLAST framework)
- Negative sampling with indirect filtering
- Complete documentation and configuration
- MIT License"
```

#### 1.3 Push to GitHub

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/DHG-LGB-Repository.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example**: If your GitHub username is `DrLiXavier`, the command would be:
```bash
git remote add origin https://github.com/DrLiXavier/DHG-LGB-Repository.git
```

---

### Part 2: Link GitHub to Zenodo (Create Permanent DOI)

#### 2.1 Sign Up for Zenodo

1. **Go to Zenodo**: Navigate to https://zenodo.org
2. **Click "Log in"**: Top-right corner
3. **Sign in with GitHub**: Click "Log in with GitHub" (easiest option)
4. **Authorize Zenodo**: Allow Zenodo to access your GitHub repositories

#### 2.2 Enable Repository in Zenodo

1. **Access Settings**: After logging in, click your username → "GitHub"
2. **Sync Repositories**: Click "Sync now" to load your GitHub repositories
3. **Find DHG-LGB-Repository**: Locate "DHG-LGB-Repository" in the list
4. **Enable Toggle**: Turn ON the toggle switch next to your repository
5. **Confirmation**: You should see "Enabled" status

#### 2.3 Create a GitHub Release (Triggers Zenodo DOI)

Back in GitHub:

1. **Navigate to Repository**: Go to https://github.com/YOUR_USERNAME/DHG-LGB-Repository
2. **Click "Releases"**: In the right sidebar
3. **Click "Create a new release"**:
   - **Tag version**: `v1.0.0` (use semantic versioning)
   - **Release title**: `DHG-LGB v1.0.0 - Initial Release`
   - **Description**:
     ```markdown
     # DHG-LGB v1.0.0 - Initial Release

     Complete implementation of Disease-Hypergraph Integrated with LightGBM framework for metabolite-disease association prediction.

     ## Features
     - HGNN model with 2-layer hypergraph neural network
     - LightGBM classifier with 5-fold cross-validation
     - All 7 evaluation metrics with confidence intervals
     - Similarity computation modules (Tanimoto, GO semantic)
     - Negative sampling with indirect association filtering
     - Comprehensive documentation and examples

     ## Citation
     If you use this code, please cite:
     [Your manuscript citation will go here]

     ## DOI
     This release is archived on Zenodo with DOI: [Auto-generated after release]
     ```
4. **Click "Publish release"**

#### 2.4 Get Your DOI from Zenodo

1. **Wait 5-10 minutes**: Zenodo automatically archives the release
2. **Check Zenodo**: Go to https://zenodo.org/account/settings/github/
3. **Find Your Repository**: You should see DHG-LGB-Repository listed
4. **Click the DOI Badge**: Copy the DOI (format: `10.5281/zenodo.XXXXXXX`)
5. **Alternative**: Go to https://zenodo.org/search?q=DHG-LGB and find your repository

#### 2.5 Add DOI Badge to README

Edit your README.md to add the Zenodo DOI badge at the top:

```markdown
# DHG-LGB: Disease-Hypergraph Integrated with LightGBM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
```

Then commit and push:

```bash
git add README.md
git commit -m "Add Zenodo DOI badge to README"
git push
```

---

### Part 3: Update Manuscript

#### 3.1 Record Your URLs and DOI

After completing the above steps, you will have:

1. **GitHub Repository URL**: `https://github.com/YOUR_USERNAME/DHG-LGB-Repository`
2. **Zenodo DOI**: `10.5281/zenodo.XXXXXXX`
3. **Zenodo URL**: `https://doi.org/10.5281/zenodo.XXXXXXX`

**Example**:
- GitHub: `https://github.com/DrLiXavier/DHG-LGB-Repository`
- DOI: `10.5281/zenodo.8234567`
- Zenodo: `https://doi.org/10.5281/zenodo.8234567`

#### 3.2 Add to Manuscript

You will need to add these to your manuscript in the **Data Availability Statement** section (see separate instructions for manuscript modifications).

---

## 🔍 Verification Checklist

After completing all steps, verify:

- [ ] GitHub repository is public and accessible
- [ ] All files are present in GitHub (check via web browser)
- [ ] README displays correctly on GitHub
- [ ] Zenodo DOI has been generated (badge visible)
- [ ] DOI link works (redirects to Zenodo archive page)
- [ ] Release v1.0.0 appears in GitHub Releases
- [ ] You have recorded GitHub URL and Zenodo DOI

---

## 📝 What to Include in Manuscript

Once you have your GitHub URL and Zenodo DOI, use this template for the **Data Availability Statement**:

```
Data Availability Statement

The source code for the DHG-LGB framework is publicly available under the MIT
License at https://github.com/[YOUR_USERNAME]/DHG-LGB-Repository. A permanent
archived version with DOI is available at https://doi.org/10.5281/zenodo.[XXXXXXX].
The repository includes all implementation code, configuration files, and
documentation required to reproduce the results presented in this study.

Raw data were obtained from publicly accessible databases: HMDB 5.0
(https://hmdb.ca, accessed 2023-03-15), CTD (https://ctdbase.org, accessed
2023-04-10), UniProt (https://www.uniprot.org), and Gene Ontology
(http://geneontology.org). Processed datasets (similarity matrices, hypergraph
structures, and training samples) are included in the code repository.
```

---

## 🆘 Troubleshooting

### Problem: "Permission denied" when pushing to GitHub

**Solution**: You need to authenticate. Use GitHub Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic) with "repo" scope
3. Copy the token
4. When prompted for password, paste the token instead

### Problem: Zenodo doesn't show my repository

**Solution**:
1. Make sure repository is **public** on GitHub
2. Click "Sync now" in Zenodo GitHub settings
3. Wait a few minutes and refresh the page
4. Check that you authorized Zenodo to access your repositories

### Problem: DOI not generated after release

**Solution**:
1. Wait 10-15 minutes (sometimes takes time)
2. Check Zenodo upload page: https://zenodo.org/deposit
3. Verify the toggle is ON in Zenodo GitHub settings
4. Try creating a new release (v1.0.1) if DOI still doesn't appear

### Problem: Git is not installed

**Solution**:
1. Download Git from https://git-scm.com/downloads
2. Install with default settings
3. Restart your terminal/command prompt
4. Verify installation: `git --version`

---

## 📞 Additional Resources

- **Git Tutorial**: https://git-scm.com/docs/gittutorial
- **GitHub Guides**: https://guides.github.com/
- **Zenodo Help**: https://help.zenodo.org/
- **Zenodo-GitHub Integration**: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content

---

## ✅ Final Notes

1. **Permanence**: Once archived on Zenodo, the DOI is **permanent** and cannot be deleted
2. **Versioning**: You can create multiple versions (v1.0.0, v1.1.0, etc.) - each gets its own DOI
3. **Citation**: The Zenodo page auto-generates citations in multiple formats (BibTeX, APA, etc.)
4. **Discoverability**: Zenodo archives are indexed by Google Scholar and other academic search engines

**After completion, you will have satisfied all reviewer requirements for code availability and scientific reproducibility!**

---

**Created**: 2025-12-07
**For**: DHG-LGB Framework Publication
