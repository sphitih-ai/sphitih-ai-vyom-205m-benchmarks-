const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const paperDir = __dirname;
const mdPath = path.join(paperDir, 'VYOM_205M_PREPRINT.md');
let content = fs.readFileSync(mdPath, 'utf8');

// Strip YAML frontmatter if present
content = content.replace(/^---[\s\S]*?---\n/, '');

// Clean up citation and figure formatting for marked
const bodyHtml = marked.parse(content);

const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vyom-205M: A Sovereign Multi-Resolution Spatio-Temporal Foundation Model</title>
<style>
  @page {
    size: letter;
    margin: 1.8cm 2.0cm;
  }
  body {
    font-family: 'Times New Roman', Times, 'Latin Modern Roman', serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #111;
    max-width: 820px;
    margin: 0 auto;
    text-align: justify;
  }
  h1.paper-title {
    font-size: 20pt;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 12px;
    line-height: 1.25;
  }
  .authors {
    text-align: center;
    font-size: 11pt;
    margin-bottom: 24px;
  }
  .authors .author-name {
    font-weight: bold;
  }
  .authors .affiliation {
    color: #444;
    font-size: 10pt;
  }
  .abstract-box {
    margin: 20px 30px 28px 30px;
    padding: 12px 18px;
    background: #fdfdfd;
    border-top: 1px solid #aaa;
    border-bottom: 1px solid #aaa;
    font-size: 9.5pt;
    line-height: 1.45;
  }
  .abstract-box strong {
    font-variant: small-caps;
    letter-spacing: 0.5px;
  }
  h1 {
    font-size: 13.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 28px;
    margin-bottom: 8px;
    border-bottom: 1.5px solid #222;
    padding-bottom: 2px;
  }
  h2 {
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 6px;
  }
  h3 {
    font-size: 10.5pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 14px;
    margin-bottom: 4px;
  }
  p {
    margin-top: 0;
    margin-bottom: 8px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 8.8pt;
  }
  th, td {
    padding: 5px 8px;
    border-top: 1px solid #ddd;
    border-bottom: 1px solid #ddd;
    text-align: left;
  }
  th {
    border-top: 1.5px solid #222;
    border-bottom: 1.5px solid #222;
    background-color: #f8f9fa;
    font-weight: bold;
  }
  tr:last-child td {
    border-bottom: 1.5px solid #222;
  }
  pre, code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.5pt;
  }
  pre {
    background: #f7f7f7;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    padding: 8px 12px;
    overflow-x: auto;
    margin: 10px 0;
    white-space: pre-wrap;
  }
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12px auto 6px auto;
    border-radius: 4px;
    border: 1px solid #ccc;
  }
  em {
    font-size: 9pt;
    color: #444;
    display: block;
    text-align: center;
    margin-bottom: 14px;
  }
</style>
</head>
<body>

<h1 class="paper-title">Vyom-205M: A Sovereign Multi-Resolution Spatio-Temporal Foundation Model for Smallholder Cadastral Delineation, SAR-to-Optical Cloud Synthesis, and Biophysical Crop Stress Diagnostics</h1>

<div class="authors">
  <div class="author-name">Sphitih AI Research Team</div>
  <div class="affiliation">Sphitih AI &nbsp;•&nbsp; <code>contact@sphitih.com</code> &nbsp;•&nbsp; <a href="https://github.com/sphitih-ai">https://github.com/sphitih-ai</a></div>
  <div class="date" style="font-size: 9.5pt; color: #666; margin-top: 4px;">November 2026</div>
</div>

<div class="abstract-box">
  <strong>Abstract</strong>—Earth observation (EO) foundation models have achieved remarkable success on coarse-resolution land-cover classification and continental-scale mapping. However, their utility deteriorates acutely across the Global South, where smallholder agricultural systems are characterized by fragmented micro-plots (&lt;0.4 ha), ultra-narrow earthen boundaries (&lt;1.5 m), persistent monsoonal cloud occlusion (&gt;75% cloud cover during peak Kharif phenology), and intricate crop-weed-stress spectral mixtures. To address these structural blindspots, we introduce <strong>Vyom-205M</strong>, a 204.96-million parameter sovereign multi-task foundation model engineered specifically for complex smallholder agrarian landscapes. Vyom-205M integrates a multi-stream hierarchical architecture that harmonizes sub-meter cadastral optical imagery, 12-band multi-temporal Sentinel-2 optical time series, multi-temporal Sentinel-1 dual-polarization synthetic aperture radar (SAR), and gridded agro-climatic covariates.
  <br/><br/>
  Evaluated across standardized international benchmarks and sovereign held-out agronomic validation sets, Vyom-205M sets new performance frontiers: (i) on the official PANGAEA / Sen1Floods11 global dual-pol flood inundation benchmark across 10 countries (30 held-out test scenes), Vyom-205M achieves <strong>75.81% Water IoU</strong>, <strong>85.83% Mean IoU</strong>, and <strong>96.33% Pixel Accuracy</strong>, surpassing IBM/NASA Prithvi-100M (68.40% Water IoU) and competing closely with Prithvi-EO-2.0-300M (77.80% Water IoU) while using 31.7% fewer parameters; (ii) on 349 sovereign smallholder cadastral validation scenes, it achieves <strong>0.8230 Parcel Extent IoU</strong> (0.9029 F1) and <strong>0.2783 2-px Relaxed Boundary F1</strong> on sub-meter earthen bunds; (iii) under the SEN12MS-CR multi-modal protocol across 116 spatio-temporal cubes, its cross-modal SAR-to-optical synthesis module achieves <strong>0.8924 SSIM</strong> and <strong>0.0298 MAE</strong>; and (iv) on 550 held-out agricultural parcels, its biophysical parcel decoder achieves <strong>58.55% Overall Accuracy</strong> and <strong>0.5730 Macro-F1</strong> across 6 diagnostic stress categories.
</div>

${bodyHtml}

</body>
</html>`;

const outHtml = path.join(paperDir, 'preprint.html');
fs.writeFileSync(outHtml, fullHtml);
console.log('Successfully generated', outHtml);
