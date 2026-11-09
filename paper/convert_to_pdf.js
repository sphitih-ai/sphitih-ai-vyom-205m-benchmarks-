const fs = require('fs');
const path = require('path');
const katex = require('katex');
const { marked } = require('marked');

const paperDir = __dirname;
const mdPath = path.join(paperDir, 'VYOM_205M_PREPRINT.md');
let content = fs.readFileSync(mdPath, 'utf8');

// Strip YAML frontmatter if present
content = content.replace(/^---[\s\S]*?---\n/, '');

// 1. Extract Display Math ($$...$$) and Inline Math ($...$) before Markdown parsing
const mathTokens = [];

// Display Math ($$...$$)
content = content.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
    const id = mathTokens.length;
    mathTokens.push({ math: math.trim(), block: true });
    return `KATEXMATHBLOCK${id}XYZ`;
});

// Inline Math ($...$) - ensure it doesn't match currency or span multiple empty lines
content = content.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    const id = mathTokens.length;
    mathTokens.push({ math: math.trim(), block: false });
    return `KATEXMATHINLINE${id}XYZ`;
});

// 2. Parse Markdown to HTML
let bodyHtml = marked.parse(content);

// 3. Render KaTeX Server-Side into HTML
bodyHtml = bodyHtml.replace(/KATEXMATH(BLOCK|INLINE)(\d+)XYZ/g, (_, type, id) => {
    const item = mathTokens[parseInt(id)];
    const isBlock = type === 'BLOCK';
    try {
        const rendered = katex.renderToString(item.math, {
            displayMode: isBlock,
            throwOnError: false
        });
        return isBlock ? `<div class="math-block">${rendered}</div>` : rendered;
    } catch (e) {
        return `<code>${item.math}</code>`;
    }
});

// 4. Correct relative image paths
bodyHtml = bodyHtml.replace(/src="figures\//g, 'src="./figures/');

// 5. Clean up captions: Turn paragraphs starting with *Figure ... into proper captions
bodyHtml = bodyHtml.replace(/<p><em>(Figure\s+\d+:[\s\S]*?)<\/em><\/p>/g, '<p class="figure-caption"><strong>$1</strong></p>');

// Full Academic HTML with precision typography
const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vyom-205M: A Sovereign Multi-Resolution Spatio-Temporal Foundation Model</title>
<!-- KaTeX Stylesheet for Fonts and Glyph Layout -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<style>
  @page {
    size: letter;
    margin: 2.2cm 2.2cm 2.2cm 2.2cm;
    @bottom-center {
      content: counter(page);
      font-size: 9pt;
      font-family: 'Times New Roman', Times, serif;
    }
  }

  *, *:before, *:after {
    box-sizing: border-box;
  }

  body {
    font-family: 'Times New Roman', Times, 'Latin Modern Roman', Georgia, serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #111;
    max-width: 820px;
    margin: 0 auto;
    text-align: justify;
    text-align-last: left; /* PREVENTS JARRING GAPS ON LAST LINES */
    text-justify: inter-word;
    word-spacing: normal;
    word-break: normal;
    overflow-wrap: break-word;
    -webkit-hyphens: auto;
    hyphens: auto;
  }

  /* Paper Title */
  h1:first-of-type {
    font-size: 18pt;
    font-weight: 700;
    text-align: center;
    text-align-last: center;
    margin-top: 5px;
    margin-bottom: 12px;
    line-height: 1.25;
    border-bottom: none;
    padding-bottom: 0;
    text-transform: none;
    letter-spacing: normal;
  }

  /* Section Headings */
  h1 {
    font-size: 12pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 24px;
    margin-bottom: 8px;
    border-bottom: 1.5px solid #222;
    padding-bottom: 2px;
    text-align: left;
    text-align-last: left;
    page-break-after: avoid;
  }

  h2 {
    font-size: 10.8pt;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 6px;
    text-align: left;
    text-align-last: left;
    page-break-after: avoid;
  }

  h3 {
    font-size: 10.2pt;
    font-weight: 700;
    font-style: italic;
    margin-top: 12px;
    margin-bottom: 4px;
    text-align: left;
    text-align-last: left;
    page-break-after: avoid;
  }

  /* Paragraphs */
  p {
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
    text-align-last: left;
    text-justify: inter-word;
  }

  /* Author block */
  p:has(strong):first-of-type {
    text-align: center;
    text-align-last: center;
    font-size: 10.5pt;
    margin-bottom: 14px;
    line-height: 1.4;
  }

  /* Abstract Box */
  h3:first-of-type {
    text-align: center;
    text-align-last: center;
    font-variant: small-caps;
    font-style: normal;
    letter-spacing: 0.8px;
    font-size: 10.5pt;
    margin-top: 10px;
    margin-bottom: 6px;
  }

  /* CRITICAL: Keep inline emphasis strictly inline! NEVER block! */
  em, i {
    font-style: italic !important;
    display: inline !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: inherit !important;
    color: inherit !important;
  }

  strong, b {
    font-weight: 700;
  }

  /* Figure Captions */
  p.figure-caption {
    font-size: 8.8pt;
    color: #333;
    text-align: justify;
    text-align-last: center;
    margin-top: 6px;
    margin-bottom: 18px;
    line-height: 1.35;
    font-style: normal;
    page-break-before: avoid;
  }

  p.figure-caption strong {
    font-weight: 600;
    color: #111;
  }

  /* Math Blocks */
  .math-block {
    text-align: center;
    margin: 10px 0;
    overflow-x: auto;
    page-break-inside: avoid;
  }

  .katex-display {
    margin: 0.5em 0 !important;
  }

  .katex {
    font-size: 1.05em;
  }

  /* Booktabs Academic Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 8.5pt;
    line-height: 1.35;
    page-break-inside: avoid;
  }

  th, td {
    padding: 5px 7px;
    border-top: 1px solid #ddd;
    border-bottom: 1px solid #ddd;
    text-align: left;
    text-align-last: left;
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
    margin: 14px auto 6px auto;
    border-radius: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    page-break-inside: avoid;
  }

  /* Ordered References List (Clean IEEE Hanging Indent) */
  ol {
    padding-left: 24px;
    margin-top: 8px;
    margin-bottom: 12px;
  }

  ol li {
    padding-left: 2px;
    margin-bottom: 4px;
    text-align: left;
    text-align-last: left;
    font-size: 9pt;
    line-height: 1.38;
  }

  ol li p {
    display: inline;
    margin: 0;
    padding: 0;
    text-align: left;
    text-align-last: left;
  }

  ul {
    padding-left: 20px;
    margin-top: 6px;
    margin-bottom: 10px;
  }

  ul li {
    margin-bottom: 4px;
    text-align: justify;
    text-align-last: left;
  }

  hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 16px 0;
  }

  pre {
    background: #f7f7f8;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 8px 12px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.2pt;
    overflow-x: auto;
    white-space: pre-wrap;
    page-break-inside: avoid;
  }

  code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.8pt;
  }
</style>
</head>
<body>

${bodyHtml}

</body>
</html>`;

const outHtml = path.join(paperDir, 'preprint.html');
fs.writeFileSync(outHtml, fullHtml);
console.log('Successfully generated preprint.html with KaTeX pre-rendering and clean typography');
