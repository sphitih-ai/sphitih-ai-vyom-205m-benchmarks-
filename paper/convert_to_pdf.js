const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const paperDir = __dirname;
const mdPath = path.join(paperDir, 'VYOM_205M_PREPRINT.md');
let content = fs.readFileSync(mdPath, 'utf8');

// Convert markdown to HTML
let bodyHtml = marked.parse(content);

// Ensure relative figure paths work properly
bodyHtml = bodyHtml.replace(/src="figures\//g, 'src="./figures/');

const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vyom-205M: A Sovereign Multi-Resolution Spatio-Temporal Foundation Model</title>
<!-- KaTeX for clean academic math rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>

<style>
  @page {
    size: letter;
    margin: 2.0cm 2.2cm;
    @bottom-center {
      content: counter(page);
      font-size: 9pt;
      font-family: 'Latin Modern Roman', 'Times New Roman', Times, serif;
    }
  }
  
  body {
    font-family: 'Latin Modern Roman', 'Times New Roman', Times, Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 820px;
    margin: 0 auto;
    text-align: justify;
    text-justify: inter-word;
    -webkit-hyphens: auto;
    hyphens: auto;
  }

  h1:first-of-type {
    font-size: 20pt;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 12px;
    line-height: 1.25;
    text-transform: none;
    letter-spacing: normal;
    border-bottom: none;
    padding-bottom: 0;
  }

  h1 {
    font-size: 12.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 28px;
    margin-bottom: 10px;
    border-bottom: 1.5px solid #222;
    padding-bottom: 3px;
    page-break-after: avoid;
  }

  h2 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }

  h3 {
    font-size: 10.5pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }

  p {
    margin-top: 0;
    margin-bottom: 10px;
    text-align: justify;
    text-justify: inter-word;
  }

  /* Author block */
  p:has(strong):first-of-type, div.authors {
    text-align: center;
    font-size: 10.5pt;
    margin-bottom: 18px;
  }

  /* Abstract Box Styling */
  h3:contains("Abstract"), h3:first-of-type {
    text-align: center;
    font-variant: small-caps;
    font-style: normal;
    letter-spacing: 1px;
    font-size: 11pt;
    margin-top: 14px;
    margin-bottom: 6px;
  }

  /* Tables (Booktabs style) */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0;
    font-size: 8.8pt;
    line-height: 1.4;
    page-break-inside: avoid;
  }

  th, td {
    padding: 6px 8px;
    border-top: 1px solid #ddd;
    border-bottom: 1px solid #ddd;
    text-align: left;
  }

  th {
    border-top: 1.5px solid #222;
    border-bottom: 1.5px solid #222;
    background-color: #f8f9fa;
    font-weight: 700;
  }

  tr:last-child td {
    border-bottom: 1.5px solid #222;
  }

  /* Figures */
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto 6px auto;
    border-radius: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    page-break-inside: avoid;
  }

  em {
    font-size: 9pt;
    color: #444;
    display: block;
    text-align: center;
    margin-top: 6px;
    margin-bottom: 18px;
    font-style: normal;
  }

  /* Numbered References List (Clean IEEE Hanging Indent) */
  ol {
    padding-left: 24px;
    margin-top: 10px;
    margin-bottom: 14px;
  }

  ol li {
    padding-left: 4px;
    margin-bottom: 7px;
    text-align: left;
    font-size: 9.5pt;
    line-height: 1.45;
  }

  hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20px 0;
  }

  pre {
    background: #f7f7f8;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 8px 12px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.5pt;
    overflow-x: auto;
    white-space: pre-wrap;
    page-break-inside: avoid;
  }

  code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
  }
</style>
</head>
<body>

${bodyHtml}

</body>
</html>`;

const outHtml = path.join(paperDir, 'preprint.html');
fs.writeFileSync(outHtml, fullHtml);
console.log('Successfully generated', outHtml);
