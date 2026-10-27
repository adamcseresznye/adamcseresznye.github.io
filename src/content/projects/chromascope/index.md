---
title: "Chromascope: A lightweight GUI for mass spectrometry data analysis"
published: 2024-10-11
draft: false
author: "Adam Cseresznye"
toc: true
---

:::note
**New article:** [Chromascope 0.3: from mzML viewer to LC-MS quantification workbench](/posts/chromascope-0-3-1) covers batch quantification, calibration, QC, projects, a command-line interface and MCP servers for AI assistants.
:::

![](chromascope_icon.png "Chromascope icon")

Have you ever needed to quickly review mass spectrometry data stored in mzML format but struggled to find software that was both fast and efficient? You’re not alone—this very challenge inspired the creation of Chromascope.

## Why Chromascope?

The mzML format, an open, XML-based standard, is widely used for mass spectrometry data storage and processing. While vendor-specific files are often easy to manage with proprietary software, finding a GUI-based tool that handles mzML files with similar ease can be difficult. Chromascope is designed to fill this gap, offering a streamlined, lightweight application for rapid data inspection.

## What’s Behind the Name?

Chromascope combines *"chromatography"* with *"scope"* evoking the sense of exploration and analysis, much like a telescope does in astronomy. The goal is to offer insightful, user-friendly interaction with mass spectrometry data.

:::note
- You can check out the project through its [github page](https://github.com/adamcseresznye/chromascope).
- The documentation can be found [here](https://adamcseresznye.github.io/chromascope/docs/chromascope/index.html).
- The binaries for Windows/Mac and Linux are downloadable [here](https://github.com/adamcseresznye/chromascope/releases/tag/v0.1.0).

:::

<img src="/projects/chromascope/demo.gif" alt="Chromascope is fast when handling complex LC-MS data" title="Title" />

## Key Features

- **mzML File Support**: Full compatibility with mzML, one of the most common open standards for mass spectrometry.
- **User-Friendly Interface**: Plot TIC, BPC, or XIC with ease by simply clicking on the chromatogram. A triple-click reveals the mass spectrum at any chosen retention time.
- **Customizable Display**: Tailor visual settings, such as smoothing, line color, and style, to suit your preferences.
- **Dark Theme Support**: For those long hours of analysis, Chromascope includes a dark mode for a more comfortable viewing experience.
- **Cross-Platform Compatibility**: Available on Windows, macOS, and Linux, Chromascope runs smoothly across multiple operating systems.

## Community Feedback and Future Improvements

To gather initial user feedback, we posted an early version of Chromascope on [Reddit](https://www.reddit.com/r/massspectrometry/comments/1f5myjn/chromascope_a_lightweight_gui_mzml_file_viewer/). The response has been overwhelmingly positive, with over 3,500 views, more than 20 comments, 13 shares, and a 100% upvote rate. This feedback has been instrumental in identifying potential areas for future enhancement, including:

- Adding support for displaying MS/MS data.
- Adding peak integration capabilities.
- Implementing background subtraction.
- Supporting additional file formats beyond mzML.

We’re excited to continue improving Chromascope, thanks to the enthusiastic feedback from the community. Let's keep in touch.
