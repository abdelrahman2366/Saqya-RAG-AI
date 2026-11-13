"""Look and feel from DESIGN.md ("Auros"): CSS tokens and components, the particle-sphere hero, and the
browser-voice player used when the online voice can't be reached. Imported by app.py."""

# Page CSS: deep-teal surfaces (abyss #012624 -> deep #011d1c -> kelp #003734), white medium-weight headings,
# silver body text, uppercase tracked labels, pink statistics, 16px cards / 6px controls, no shadows, the aurora
# gradient only on the send button.
BASE_CSS = """<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500&display=swap");
[data-testid="stMarkdownContainer"], [data-testid="stChatInput"] textarea, [data-testid="stWidgetLabel"],
[data-testid="stBaseButton-secondary"] p, [data-testid="stButtonGroup"] p {
  font-family: Inter, "IBM Plex Sans Arabic", "Segoe UI", system-ui, sans-serif;}
:root {
  --abyss:#012624; --deep:#011d1c; --kelp:#003734; --mist:#edfffe; --platinum:#ffffff; --silver:#bbc7c6;
  --slate:#707777; --phosphor:#fde9ff; --line:rgba(255,255,255,.10); --teal-ghost:rgba(3,81,75,.5);
  --aurora:linear-gradient(90deg, rgb(203,255,252) 0%, rgb(237,255,254) 26.25%, rgb(255,253,250) 47.57%, rgb(250,209,255) 88.96%);
  --biolum:linear-gradient(90deg, rgb(0,130,124) 0%, rgb(203,255,252) 100%);
}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stMainBlockContainer"] {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 6rem;}
h1, h2, h3, h4 {color: var(--platinum) !important; font-weight: 500 !important; letter-spacing: -0.02em;}
*, *::before, *::after {box-shadow: none !important;}

/* ---- type */
.aa-eyebrow {font-size: 12px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--silver);}
.aa-eyebrow.small {font-size: 10px; letter-spacing: .15em;}
.aa-title {font-size: clamp(2.1rem, 4.2vw, 3rem); line-height: 1; letter-spacing: -0.04em; color: var(--platinum);
           font-weight: 500; margin: .55rem 0 .7rem;}
.aa-sub {color: var(--silver); font-size: 16px; line-height: 1.45; max-width: 640px; margin: 0;}
.aa-section {margin: 2.6rem 0 1rem;}

/* ---- header */
.aa-brand {display: flex; align-items: center; gap: 12px; height: 40px;}
.aa-mark {width: 30px; height: 30px; flex: none; display: inline-flex;}
.aa-wheel {animation: aa-turn 24s linear infinite; transform-origin: 16px 13.6px; transform-box: view-box;}
@keyframes aa-turn {to {transform: rotate(360deg);}}
@media (prefers-reduced-motion: reduce) {.aa-wheel {animation: none;}}
.aa-word-sep {color: var(--silver); font-size: 14px; margin: 0 -4px;}
.aa-word-ar {font-family: "IBM Plex Sans Arabic", Inter, sans-serif; font-size: 18px; font-weight: 500; color: var(--platinum);
             letter-spacing: 0; line-height: 1;}
/* Arabic inside tracked (letter-spaced) labels: tracking would break the joined letters */
.ar, [lang="ar"] {letter-spacing: 0 !important; text-transform: none !important;}
.aa-word {font-size: 14px; font-weight: 500; letter-spacing: .15em; text-transform: uppercase; color: var(--platinum);}
.aa-dot {display: inline-flex; align-items: center; gap: 7px; font-size: 10px; letter-spacing: .15em;
         text-transform: uppercase; color: var(--silver); border: 1px solid var(--line); border-radius: 6px;
         padding: 4px 9px; white-space: nowrap;}
.aa-dot::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--slate);}
.aa-dot.ok::before {background: #cbfffc;}
.aa-dot.degraded::before {background: transparent; border: 1.5px solid var(--mist);}

/* segmented controls (view + language) read as ghost navigation links */
[data-testid="stButtonGroup"] {gap: 4px;}
[data-testid="stButtonGroup"] button {background: transparent !important; border: none !important; border-radius: 6px;
   color: var(--silver) !important; text-transform: uppercase; letter-spacing: .12em; font-size: 12px; padding: 6px 10px;}
[data-testid="stButtonGroup"] button p {font-size: 12px; font-weight: 500;}
[data-testid="stButtonGroup"] button[aria-checked="true"] {
   color: var(--platinum) !important; background: var(--teal-ghost) !important;}
[data-testid="stButtonGroup"] button:hover {color: var(--platinum) !important;}

/* ---- statistics */
.aa-stats {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 20px; margin: 8px 0 0;}
.aa-stat-num {color: var(--phosphor); font-size: clamp(2rem, 4.6vw, 3.3rem); font-weight: 500; line-height: 1;
              letter-spacing: -0.04em; font-variant-numeric: tabular-nums;}
.aa-stat-label {color: var(--mist); font-size: 12px; letter-spacing: .1em; text-transform: uppercase; margin-top: 10px;}
@media (max-width: 640px) {.aa-stats {grid-template-columns: repeat(2, minmax(0, 1fr));}}

/* ---- buttons */
[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {
   background: var(--kelp) !important; border: 1px solid transparent !important; color: var(--mist) !important;}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {
   border-color: rgba(203,255,252,.35) !important; color: var(--platinum) !important;}
[data-testid="stBaseButton-primary"] {background: var(--aurora) !important; color: #222222 !important; border: none !important;}
[data-testid="stBaseButton-primary"] p {font-size: 13px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase;}
[data-testid="stBaseButton-tertiary"] {color: var(--silver) !important;}
[data-testid="stBaseButton-tertiary"]:hover {color: var(--platinum) !important;}

/* suggestion cards: feature-row cards with the 32x32 arrow button */
.st-key-suggest [data-testid="stBaseButton-secondary"] {width: 100%; min-height: 96px; border-radius: 16px; padding: 18px 58px 18px 20px;
   justify-content: flex-start; text-align: start; position: relative; white-space: normal;}
.st-key-suggest [data-testid="stBaseButton-secondary"] p {font-size: 15px; line-height: 1.4; text-align: start;}
.st-key-suggest [data-testid="stBaseButton-secondary"] * {white-space: normal !important; overflow: visible !important;
   text-overflow: clip !important;}
.st-key-suggest [data-testid="stBaseButton-secondary"]::after {content: "↗"; position: absolute; top: 14px; right: 14px;
   width: 32px; height: 32px; border-radius: 6px; background: var(--teal-ghost); color: var(--platinum);
   display: flex; align-items: center; justify-content: center; font-size: 15px;}
.st-key-suggest .aa-eyebrow {margin: 0 0 10px;}

/* ---- chat */
[data-testid="stChatMessage"] {background: transparent; gap: 14px; padding: 0; margin-bottom: 14px;}
[data-testid="stChatMessage"]:has(.role-assistant) {background: var(--kelp); border-radius: 16px; padding: 24px 28px;}
[data-testid="stChatMessage"]:has(.role-user) {border: 1px solid var(--line); border-radius: 16px; padding: 16px 24px; align-items: center;}
[data-testid="stChatMessage"] > div:first-child:not([data-testid="stChatMessageContent"]) {
   background: var(--teal-ghost) !important; border-radius: 6px !important; width: 32px; height: 32px; font-size: 17px;}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {color: var(--mist); font-size: 16px; line-height: 1.6;}
.cite {display: inline-block; min-width: 1.5em; padding: 0 .38em; margin: 0 .12em; border-radius: 6px; background: var(--teal-ghost);
       color: var(--mist); font-size: .72em; line-height: 1.55; text-align: center; vertical-align: .18em; unicode-bidi: isolate;
       font-variant-numeric: tabular-nums;}
.aa-badges {display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px;}
.aa-badge {font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--silver); border: 1px solid var(--line);
           border-radius: 6px; padding: 3px 8px; unicode-bidi: isolate;}
.aa-badge.alert {color: var(--abyss); background: var(--mist); border-color: var(--mist);}
.aa-badge.web, .aa-badge.learned {color: var(--mist); border-color: rgba(237,255,254,.35);}
.aa-badge.general {color: var(--phosphor); border-color: rgba(253,233,255,.45);}
.aa-hint {color: var(--silver); font-size: 14px; margin: 6px 0 10px;}
.source-chip {display: inline-block; margin: 4px 6px 0 0; padding: 4px 10px; border-radius: 6px; background: var(--deep);
              color: var(--silver); font-size: 12.5px; unicode-bidi: isolate;}
a.source-chip {text-decoration: none; color: var(--mist) !important; border: 1px solid transparent;}
a.source-chip:hover {border-color: var(--line); color: var(--platinum) !important;}
a.source-chip .site {color: var(--silver); margin-inline-start: 6px;}
.aa-passage-meta a {color: var(--mist) !important; text-decoration: underline; text-decoration-color: var(--line);}
.aa-saved {color: var(--slate); font-size: 12px; margin-top: 10px;}
/* ---- while the answer is being found: words that take turns beside three pulsing dots */
.aa-thinking {display: flex; align-items: center; gap: 12px; min-height: 30px; color: var(--silver); font-size: 15px;}
.aa-dots {display: inline-flex; gap: 5px;}
.aa-dots i {width: 7px; height: 7px; border-radius: 50%; background: var(--mist); opacity: .25; animation: aa-pulse 1.2s infinite ease-in-out;}
.aa-dots i:nth-child(2) {animation-delay: .2s;} .aa-dots i:nth-child(3) {animation-delay: .4s;}
@keyframes aa-pulse {0%, 80%, 100% {opacity: .25; transform: scale(.85);} 40% {opacity: 1; transform: scale(1);}}
.aa-words {position: relative; display: inline-block; height: 1.5em; min-width: 18ch; overflow: hidden;}
.aa-words b {position: absolute; inset-inline-start: 0; top: 0; font-weight: 400; white-space: nowrap; opacity: 0;
             animation-name: aa-word; animation-iteration-count: infinite; animation-timing-function: ease;}
.aa-words b:first-child {animation-name: aa-word-first;}
@keyframes aa-word {0% {opacity: 0; transform: translateY(45%);} 3% {opacity: 1; transform: none;}
                    14% {opacity: 1; transform: none;} 17% {opacity: 0; transform: translateY(-45%);} 100% {opacity: 0;}}
/* the first word is there at once (no fade-in), so the box is never empty */
@keyframes aa-word-first {0% {opacity: 1; transform: none;} 14% {opacity: 1; transform: none;}
                          17% {opacity: 0; transform: translateY(-45%);} 97% {opacity: 0; transform: translateY(45%);} 100% {opacity: 1; transform: none;}}
.aa-sources-label {font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--silver); margin-top: 14px;}
[data-testid="stChatMessage"] [data-testid="stExpander"] details {border: 1px solid var(--line); border-radius: 6px; background: transparent;}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary {background: transparent !important;}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary:hover p {color: var(--platinum);}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary p {font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--silver);}
.aa-passage {background: var(--deep); border-radius: 6px; padding: 14px 16px; margin: 0 0 10px;}
.aa-passage-meta {display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; font-size: 11px; letter-spacing: .08em;
                  text-transform: uppercase; color: var(--silver); margin-bottom: 8px;}
.aa-passage-meta b {color: var(--platinum); font-weight: 500;}
.aa-score {height: 3px; border-radius: 2px; background: rgba(255,255,255,.08); margin: 0 0 10px; overflow: hidden;}
.aa-score > span {display: block; height: 100%; background: var(--biolum);}
.aa-passage-text {color: var(--mist); font-size: 14px; line-height: 1.6; white-space: pre-wrap; unicode-bidi: plaintext; text-align: start;}
.aa-passage-meta .file {text-transform: none; letter-spacing: 0; font-size: 12px;}
.aa-focus {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}

/* chat input: a kelp card with the aurora send button (the page's one primary action) */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {background: var(--abyss);}
[data-testid="stChatInput"] {background: var(--kelp) !important; border: 1px solid var(--line) !important; border-radius: 16px !important;}
[data-testid="stChatInput"] textarea {color: var(--mist) !important; unicode-bidi: plaintext; text-align: start;}
[data-testid="stChatInputSubmitButton"] {background: var(--aurora) !important; border-radius: 6px !important; color: #222222 !important;}
[data-testid="stChatInputSubmitButton"] svg {fill: #222222 !important; color: #222222 !important;}

/* ---- surfaces */
.st-key-crops [data-testid="stVerticalBlock"] [class*="st-key-crop-"] {background: var(--kelp); border-radius: 16px; padding: 22px 22px 16px;
   min-height: 196px; justify-content: space-between;}
.aa-crop-ar {font-size: 24px; line-height: 1.2; color: var(--platinum); font-weight: 500;}
.aa-crop-en {font-size: 11px; letter-spacing: .15em; text-transform: uppercase; color: var(--silver); margin: 6px 0 14px;}
.aa-crop-meta {font-size: 13px; color: var(--silver); margin-bottom: 4px;}
.aa-card {background: var(--kelp); border-radius: 16px; padding: 32px; height: 100%;}
.aa-card h4 {font-size: 24px; margin: 14px 0 10px; letter-spacing: -0.02em;}
.aa-card p {color: var(--silver); font-size: 15px; line-height: 1.5; margin: 0;}
.aa-step {font-size: 12px; letter-spacing: .12em; color: var(--phosphor); font-variant-numeric: tabular-nums;}
.aa-well {background: var(--deep); border-radius: 16px; padding: 36px 32px; margin-top: 3rem; color: var(--silver); font-size: 13px; line-height: 1.6;}

/* ---- sidebar */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--silver);}
[data-testid="stSidebar"] .aa-status {background: var(--kelp); border-radius: 16px; padding: 18px 18px 14px; margin-bottom: 10px;}
[data-testid="stSidebar"] .aa-status-title {font-size: 18px; color: var(--platinum); font-weight: 500; margin: 8px 0 4px;}
[data-testid="stSidebar"] .aa-status-detail {font-size: 13px; color: var(--silver); line-height: 1.45;}
[data-testid="stSidebar"] .aa-disclaimer {font-size: 12px; color: var(--slate); line-height: 1.5; margin-top: 18px;}
[data-testid="stAlert"] {border-radius: 6px;}

/* ---- uploaded document */
[data-testid="stFileUploaderDropzone"] {background: var(--kelp) !important; border: 1px dashed rgba(203,255,252,.35) !important;
   border-radius: 16px !important; padding: 28px !important;}
[data-testid="stFileUploaderDropzone"] button {background: var(--deep) !important; color: var(--mist) !important; border: none !important;}
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small {color: var(--silver) !important;}
.aa-doc {background: var(--kelp); border-radius: 16px; padding: 26px 28px; margin: 6px 0 4px;}
.aa-doc-title {font-size: 26px; line-height: 1.2; color: var(--platinum); font-weight: 500; margin: 10px 0 8px; letter-spacing: -0.02em;}
.aa-doc-meta {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px;}
.aa-doc-file {color: var(--silver); font-size: 13px; unicode-bidi: plaintext;}
.st-key-sections [data-testid="stBaseButton-secondary"] {min-height: 0; padding: 6px 12px; border-radius: 6px;}
.st-key-sections [data-testid="stBaseButton-secondary"] p {font-size: 13px;}
.st-key-recent [data-testid="stBaseButton-secondary"] {justify-content: flex-start; text-align: start;}
.st-key-recent [data-testid="stBaseButton-secondary"] p {font-size: 14px; text-align: start;}

/* ---- the attached document, pinned above the conversation */
.st-key-docbar {background: var(--kelp); border-radius: 16px; padding: 14px 16px 14px 18px; border: 1px solid rgba(203,255,252,.14);}
.st-key-docbar [data-testid="stMarkdownContainer"] {min-width: 0;}
.st-key-docbar > div:first-child {flex: 1 1 auto; min-width: 0;}
.aa-docbar {display: flex; align-items: center; gap: 14px; min-width: 0;}
.aa-docbar-icon {font-size: 26px; line-height: 1; flex: none;}
.aa-docbar-text {min-width: 0;}
.aa-docbar-title {font-size: 18px; color: var(--platinum); font-weight: 500; line-height: 1.3; margin: 4px 0 4px;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; unicode-bidi: plaintext;}
.aa-docbar-meta {display: flex; flex-wrap: wrap; gap: 6px; color: var(--silver); font-size: 12px; line-height: 1.5;}
.aa-docbar-meta .file {color: var(--mist);}
.st-key-docsuggest {margin: 12px 0 4px; flex-wrap: wrap;}
.st-key-docsuggest [data-testid="stBaseButton-secondary"] {min-height: 0; padding: 7px 14px; border-radius: 6px;}
.st-key-docsuggest [data-testid="stBaseButton-secondary"] p {font-size: 13px;}

/* ---- settings */
.st-key-set-voice, .st-key-set-conv, .st-key-set-sources, .st-key-set-learned {background: var(--kelp); border-radius: 16px;
   padding: 26px 26px 22px; height: 100%;}
.st-key-set-sources [data-testid="stCheckbox"] label p, .st-key-set-sources [data-testid="stToggle"] label p {color: var(--mist); font-size: 14px;}
/* the switch's thumb slides left-to-right; in the Arabic page it would slide out of its track */
[data-testid="stCheckbox"] label > div:first-of-type {direction: ltr; flex-shrink: 0;}
.aa-learned-row {display: flex; gap: 10px; align-items: baseline; padding: 8px 0; border-top: 1px solid var(--line); font-size: 13px;}
.aa-learned-row .kind {font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--silver); white-space: nowrap;}
.aa-learned-row .what {color: var(--mist); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.aa-learned-row .what a {color: var(--mist) !important; text-decoration: none;}
.aa-learned-row .n {color: var(--slate); white-space: nowrap; font-variant-numeric: tabular-nums;}
.st-key-set-learned [data-testid="stBaseButton-tertiary"] {min-height: 0; padding: 2px 6px;}
.st-key-set-learned [data-testid="stBaseButton-tertiary"] p {font-size: 12px; color: var(--silver);}
.st-key-set-learned [data-testid="stBaseButton-secondary"] {background: var(--abyss) !important; border: 1px solid var(--line) !important;}
.aa-set-title {font-size: 22px; color: var(--platinum); font-weight: 500; letter-spacing: -0.02em;}
.aa-set-sub {color: var(--silver); font-size: 14px; margin: 6px 0 14px;}
.aa-set-count {color: var(--mist); font-size: 13px; margin: 0 0 12px;}
.st-key-set-conv [data-testid="stBaseButton-secondary"], .st-key-set-conv [data-testid="stDownloadButton"] button {
   background: var(--abyss) !important; border: 1px solid var(--line) !important;}
.aa-badge.file {text-transform: none; letter-spacing: 0;}

/* ---- voice */
.aa-heard {color: var(--silver); font-size: 13px; margin-top: 6px; unicode-bidi: plaintext;}
.aa-hint-line {color: var(--silver); font-size: 14px; margin: 18px 0 0;}
[data-testid="stAudio"] audio, audio {color-scheme: dark; width: 100%; height: 38px;}
.aa-voice-note {font-size: 12px; color: var(--slate); line-height: 1.5; margin-top: 6px;}

/* Each chat message carries an invisible .lang-ar / .lang-en marker; :has() sets its direction, so
   Arabic answers read right-to-left in the English UI too (and the other way round). */
[data-testid="stChatMessage"]:has(.lang-ar) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has(.lang-ar) [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {direction: rtl; text-align: right !important;}
[data-testid="stChatMessage"]:has(.lang-en) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has(.lang-en) [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {direction: ltr; text-align: left !important;}
/* tracking breaks the joined letters of Arabic script: Arabic text is never letter-spaced */
[lang="ar"], .aa-passage-meta bdi:not(.file) {letter-spacing: 0 !important; text-transform: none !important;}
.lang-ar, .lang-en, .role-user, .role-assistant {display: none;}
</style>"""

# Extra rules for the Arabic (right-to-left) page. Arabic text is never letter-spaced: tracking breaks the
# joined letters.
RTL_CSS = """<style>
  [data-testid="stMainBlockContainer"], [data-testid="stSidebar"] {direction: rtl;}
  [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {text-align: right;}
  [data-testid="stChatInput"] textarea {direction: rtl;}
  .st-key-suggest [data-testid="stBaseButton-secondary"] {padding: 18px 20px 18px 58px;}
  .st-key-suggest [data-testid="stBaseButton-secondary"]::after {right: auto; left: 14px; content: "↖";}
  .source-chip {margin: 4px 0 0 6px;}
  .aa-eyebrow, .aa-badge, .aa-stat-label, .aa-dot, .aa-sources-label, .aa-crop-en, .aa-step,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stButtonGroup"] button,
  [data-testid="stExpander"] summary p {letter-spacing: 0 !important;}
  .aa-title {letter-spacing: 0; line-height: 1.2;}
  .aa-doc-title {letter-spacing: 0;}
  .st-key-recent [data-testid="stBaseButton-secondary"] {justify-content: flex-start;}
</style>"""

# DESIGN.md "Particle Sphere Visual": a rotating sphere of teal / aqua / pink dots behind the hero text.
# __DIR__, __EYEBROW__, __TITLE__, __SUB__ and __STATS__ are filled in by app.py.
HERO_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=IBM+Plex+Sans+Arabic:wght@400;500&display=swap" rel="stylesheet">
<style>
  html, body {margin:0; height:100%; background:#012624; overflow:hidden;
              font-family: Inter, "IBM Plex Sans Arabic", system-ui, sans-serif; -webkit-font-smoothing: antialiased;}
  canvas {position:absolute; inset:0; width:100%; height:100%;}
  .wrap {position:relative; z-index:1; height:100%; display:flex; flex-direction:column; justify-content:center; padding:0 4px;}
  .eyebrow {font-size:12px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:#bbc7c6;}
  .ar {letter-spacing:0; font-family:"IBM Plex Sans Arabic", Inter, sans-serif;}
  h1 {font-size:clamp(40px, 6.4vw, 76px); line-height:1; letter-spacing:-0.045em; font-weight:500; color:#fff;
      margin:16px 0 18px; max-width:11ch;}
  [dir=rtl] h1 {letter-spacing:0; line-height:1.15;}
  [dir=rtl] .eyebrow, [dir=rtl] .lab {letter-spacing:0;}
  p {color:#bbc7c6; font-size:16px; line-height:1.5; max-width:min(520px, 50%); margin:0;}
  .stats {display:flex; flex-wrap:wrap; gap:18px 44px; margin-top:38px;}
  .num {color:#fde9ff; font-size:clamp(30px, 4vw, 46px); font-weight:500; letter-spacing:-0.04em; line-height:1;
        font-variant-numeric: tabular-nums;}
  .lab {color:#edfffe; font-size:11px; letter-spacing:.12em; text-transform:uppercase; margin-top:9px;}
  @media (max-width: 640px) { p {max-width:none;} .stats {gap:16px 28px;} }
</style></head><body>
<canvas id="orb"></canvas>
<div class="wrap" dir="__DIR__">
  <div class="eyebrow">__EYEBROW__</div>
  <h1>__TITLE__</h1>
  <p>__SUB__</p>
  <div class="stats">__STATS__</div>
</div>
<script>
(() => {
  const rtl = "__DIR__" === "rtl";
  const cv = document.getElementById("orb"), ctx = cv.getContext("2d");
  const N = 2200, pts = [], golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) {           // Fibonacci sphere: evenly spread points
    const y = 1 - (i / (N - 1)) * 2, r = Math.sqrt(1 - y * y), t = golden * i;
    pts.push([Math.cos(t) * r, y, Math.sin(t) * r, Math.random()]);
  }
  let w = 0, h = 0, dpr = 1;
  function size() {
    dpr = window.devicePixelRatio || 1; w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  const mix = (a, b, k) => a.map((v, i) => Math.round(v + (b[i] - v) * k));
  const TEAL = [0, 130, 124], AQUA = [203, 255, 252], PINK = [250, 209, 255];
  function frame(time) {
    const narrow = w < 700, R = narrow ? Math.min(w, h) * 0.46 : Math.min(h * 0.43, w * 0.25);
    const cx = narrow ? w * 0.5 : (rtl ? w * 0.24 : w * 0.76), cy = h * 0.5;
    const a = time * 0.00012, tilt = 0.42, ca = Math.cos(a), sa = Math.sin(a), ct = Math.cos(tilt), st = Math.sin(tilt);
    ctx.clearRect(0, 0, w, h);
    for (const [x0, y0, z0, n] of pts) {
      const x1 = x0 * ca + z0 * sa, z1 = -x0 * sa + z0 * ca;          // spin around the vertical axis
      const y2 = y0 * ct - z1 * st, z2 = y0 * st + z1 * ct;           // tilt towards the viewer
      const breathe = 1 + 0.018 * Math.sin(time * 0.0011 + n * 6.28);
      const p = 2.6 / (2.6 - z2), px = cx + x1 * R * p * breathe, py = cy + y2 * R * p * breathe;
      const depth = (z2 + 1) / 2, rim = Math.pow(1 - Math.abs(z2), 3);
      let col = mix(TEAL, AQUA, depth);
      col = mix(col, PINK, Math.min(1, rim * 1.6 * (n > 0.5 ? 1 : 0.4)));
      const alpha = (narrow ? 0.35 : 0.9) * (0.12 + depth * 0.78);
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha.toFixed(3)})`;
      const s = 0.5 + depth * 1.3;
      ctx.beginPath(); ctx.arc(px, py, s, 0, 6.2832); ctx.fill();
    }
  }
  size(); window.addEventListener("resize", size);
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const loop = (t) => { frame(t); if (!still) requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
})();
</script></body></html>"""

# Plays an answer with the browser's own speech synthesis (no internet needed). Browsers only speak after a
# click, hence the button. __TEXT__ is a JSON string; __LANG__ "ar-EG" or "en-US".
# ---------------------------------------------------------------- live voice
# The answer is spoken live, sentence by sentence: each sentence is an <audio> element pointing at the
# backend's GET /voice/stream, which sends the MP3 while the voice service is still making it, so the first
# words play within a moment; the next sentence loads while the current one plays. Without the online voice
# (no internet, or the browser can't reach the backend) the browser's own voice reads the sentences instead.
# The player is put into the Streamlit page itself (the components' iframes are same-origin), so a rerun of
# the app — a rating, a new question being typed — doesn't cut it off. It stops when the mic is pressed, and
# when its message is no longer on screen.
PLAYER_VERSION = "2"
PLAYER_JS = r"""(function () {
  var old = window.__saqyaVoice;
  if (old && old.version === "__VERSION__") return;
  try { if (old) old.stop(); } catch (e) {}
  var v = {version: "__VERSION__", token: "", sentences: [], i: -1, state: "idle", engine: "server", lang: "ar",
           spoken: old ? old.spoken : {}, seen: 0};
  var cur = null, next = null, url = "", voice = "female", style = "egyptian", engine = "edge", run = 0;
  var synth = window.speechSynthesis || null;

  function src(k) {
    return url + "?" + new URLSearchParams({text: v.sentences[k], language: v.lang, voice: voice, style: style,
                                            engine: engine}).toString();
  }
  function load(k) { var a = new Audio(); a.preload = "auto"; a._k = k; a.src = src(k); return a; }
  function drop(a) {
    if (!a) return;
    a.onplaying = a.onended = a.onerror = null;
    try { a.pause(); a.removeAttribute("src"); a.load(); } catch (e) {}
  }
  function halt() { run++; drop(cur); drop(next); cur = next = null; if (synth) { try { synth.cancel(); } catch (e) {} } }
  function finished() { v.state = "done"; v.i = -1; }

  function server(k, r) {
    if (r !== run) return;
    if (k >= v.sentences.length) return finished();
    v.i = k;
    if (v.state !== "speaking") v.state = "loading";
    var a = (next && next._k === k) ? next : load(k);
    if (next && next !== a) drop(next);
    next = null;
    if (cur && cur !== a) drop(cur);
    cur = a;
    var failed = false;
    function fail() {
      if (failed || r !== run) return;
      failed = true;
      v.engine = "browser";          // online voice unreachable: the browser reads the rest
      drop(next); next = null;
      browser(k, r);
    }
    a.onplaying = function () {
      if (r !== run) return;
      v.state = "speaking";
      if (!next && k + 1 < v.sentences.length) next = load(k + 1);   // load the next sentence now
    };
    a.onended = function () { server(k + 1, r); };
    a.onerror = fail;
    var p = a.play();
    if (p && p.catch) p.catch(function (err) {
      if (r !== run) return;
      if (err && err.name === "NotAllowedError") v.state = "blocked";  // the browser wants a click first
      else if (err && err.name !== "AbortError") fail();
    });
  }

  function pickVoice() {
    var list = synth.getVoices() || [], want = v.lang === "ar" ? "ar" : "en";
    var exact = want === "ar" ? "ar-eg" : "en-us";
    return list.find(function (x) { return x.lang && x.lang.toLowerCase().replace("_", "-").indexOf(exact) === 0; })
        || list.find(function (x) { return x.lang && x.lang.toLowerCase().indexOf(want) === 0; }) || null;
  }
  function browser(k, r) {
    if (r !== run) return;
    if (!synth) { v.state = "novoice"; return; }
    if (k >= v.sentences.length) return finished();
    v.i = k;
    v.state = "speaking";
    var u = new SpeechSynthesisUtterance(v.sentences[k]);
    u.lang = v.lang === "ar" ? "ar-EG" : "en-US";
    var chosen = pickVoice();
    if (chosen) u.voice = chosen;
    else if (v.lang === "ar" && synth.getVoices().length) { v.state = "novoice"; return; }
    u.rate = 0.95;
    u.onend = function () { browser(k + 1, r); };
    u.onerror = function (e) { if (r === run && e && e.error === "not-allowed") v.state = "blocked"; };
    synth.speak(u);
  }

  v.start = function (json) {
    var c = JSON.parse(json);
    halt();
    v.token = c.token; v.sentences = c.sentences; v.lang = c.lang; url = c.url || ""; voice = c.voice || "female";
    style = c.style || "egyptian"; engine = c.engine || "edge";
    v.engine = url ? "server" : "browser"; v.i = 0; v.state = "loading"; v.spoken[c.token] = true; v.seen = Date.now();
    (v.engine === "server" ? server : browser)(0, run);
  };
  v.pause = function () {
    if (v.state !== "speaking" && v.state !== "loading") return;
    if (v.engine === "server" && cur) cur.pause();
    else { run++; if (synth) synth.cancel(); }   // browser voices pause unreliably: stop, resume repeats the sentence
    v.state = "paused";
  };
  v.resume = function () {
    var k = Math.max(v.i, 0);
    if (v.state === "paused" && v.engine === "server" && cur) {
      v.state = "speaking";
      var p = cur.play();
      if (p && p.catch) p.catch(function () {});
    } else if (v.state === "paused" || v.state === "blocked") {
      halt();
      v.state = "loading";
      (v.engine === "server" ? server : browser)(k, run);
    }
  };
  v.stop = function () { halt(); v.state = "stopped"; v.i = -1; };

  window.__saqyaVoice = v;
  if (!window.__saqyaVoiceHooks) {
    window.__saqyaVoiceHooks = true;
    // pressing the microphone stops the voice, so the recording doesn't pick it up
    document.addEventListener("click", function (e) {
      var t = e.target;
      if (t && t.closest && t.closest('[data-testid="stChatInputMicButton"]') && window.__saqyaVoice) window.__saqyaVoice.stop();
    }, true);
    // the message being read is gone (another page of the app): stop talking
    setInterval(function () {
      var w = window.__saqyaVoice;
      if (w && (w.state === "speaking" || w.state === "loading") && Date.now() - w.seen > 4000) w.stop();
    }, 1000);
  }
})();"""

LIVE_VOICE_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html, body {margin:0; background:transparent; overflow:hidden;
              font-family: Inter, "IBM Plex Sans Arabic", system-ui, sans-serif;}
  .bar {display:flex; align-items:center; gap:10px; height:44px; margin:4px 1px; padding:0 8px 0 6px;
        box-sizing:border-box; border:1px solid rgba(203,255,252,.16); border-radius:10px; background:rgba(203,255,252,.04);}
  button {flex:none; width:32px; height:32px; padding:0; border:none; border-radius:8px; cursor:pointer;
          display:grid; place-items:center;}
  button svg {width:14px; height:14px;}
  .main {background:linear-gradient(90deg, rgb(203,255,252) 0%, rgb(237,255,254) 26.25%, rgb(255,253,250) 47.57%,
         rgb(250,209,255) 88.96%); color:#0d2626;}
  .ghost {background:transparent; color:#bbc7c6; border:1px solid rgba(203,255,252,.2);}
  .ghost:hover {color:#fff; border-color:rgba(203,255,252,.45);}
  .eq {flex:none; display:flex; align-items:flex-end; gap:2px; height:16px;}
  .eq i {display:block; width:3px; height:4px; border-radius:2px; background:#cbfffc; opacity:.45;}
  .eq i:nth-child(2) {background:#e3fffd; animation-delay:-.2s;}
  .eq i:nth-child(3) {background:#fffdfa; animation-delay:-.45s;}
  .eq i:nth-child(4) {background:#f3dcff; animation-delay:-.65s;}
  .eq i:nth-child(5) {background:#fad1ff; animation-delay:-.3s;}
  .speaking .eq i {opacity:1; animation:eq .9s ease-in-out infinite;}
  .loading .eq i {opacity:.8; animation:eq 1.6s ease-in-out infinite;}
  @keyframes eq {0%, 100% {height:4px;} 50% {height:16px;}}
  .cap {flex:1; min-width:0; display:flex; align-items:baseline; gap:8px; font-size:13px; color:#e7f1f0;}
  .label {flex:none; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#8fa8a6;}
  [dir=rtl] .label {letter-spacing:0;}
  .text {flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .idle .text, .done .text, .stopped .text {color:#8fa8a6;}
  .blocked .label {color:#fad1ff;}
  .count {flex:none; font-size:11px; color:#8fa8a6; font-variant-numeric:tabular-nums;}
  .hide {display:none;}
  @media (prefers-reduced-motion: reduce) {.speaking .eq i, .loading .eq i {animation:none; height:10px;}}
</style></head><body>
<div class="bar idle" id="bar" dir="__DIR__">
  <button class="main" id="toggle"></button>
  <div class="eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="cap"><span class="label" id="label"></span><span class="text" id="text" dir="auto"></span></div>
  <span class="count" id="count" dir="ltr"></span>
  <button class="ghost hide" id="stop"></button>
</div>
<script>
  const CFG = __CONFIG__;
  const T = CFG.t;
  const ICON = {
    play: '<svg viewBox="0 0 16 16"><path d="M4.5 2.5v11l9-5.5z" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 16 16"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>',
    stop: '<svg viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="currentColor"/></svg>',
    again: '<svg viewBox="0 0 16 16"><path d="M8 2.5a5.5 5.5 0 1 1-5.2 3.7l1.4.5A4 4 0 1 0 8 4v2L4.8 3.2 8 .5z" fill="currentColor"/></svg>',
  };
  // The player lives in the app's page (same origin), so it keeps talking when Streamlit redraws this frame
  function player() {
    let host = window;
    try { if (window.parent && window.parent.document) host = window.parent; } catch (e) {}
    if (!host.__saqyaVoice || host.__saqyaVoice.version !== CFG.version) {
      const s = host.document.createElement("script");
      s.textContent = CFG.player.split("__VERSION__").join(CFG.version);
      host.document.head.appendChild(s);
    }
    return host.__saqyaVoice;
  }
  const v = player();
  const job = () => JSON.stringify({token: CFG.token, sentences: CFG.sentences, lang: CFG.lang, url: CFG.url, voice: CFG.voice,
                                    style: CFG.style, engine: CFG.engine});
  if (v && CFG.autoplay && !v.spoken[CFG.token]) v.start(job());   // once per answer; later only on request

  const $ = (id) => document.getElementById(id);
  const n = CFG.sentences.length;
  function render() {
    const mine = v && v.token === CFG.token;
    const st = mine ? v.state : "idle";
    if (mine) v.seen = Date.now();
    $("bar").className = "bar " + st;
    const playing = st === "speaking" || st === "loading";
    $("toggle").innerHTML = playing ? ICON.pause : (st === "paused" || st === "blocked") ? ICON.play : ICON.again;
    $("toggle").title = playing ? T.pause : (st === "paused" || st === "blocked") ? T.play : T.again;
    $("stop").innerHTML = ICON.stop; $("stop").title = T.stop;
    $("stop").classList.toggle("hide", !(playing || st === "paused"));
    const label = {speaking: T.speaking, loading: T.loading, paused: T.paused, blocked: T.blocked, novoice: T.novoice}[st] || T.again;
    $("label").textContent = label + (mine && v.engine === "browser" && playing ? " · " + T.browser : "");
    const i = mine && v.i >= 0 ? v.i : 0;
    $("text").textContent = st === "blocked" || st === "novoice" ? "" : CFG.sentences[i];
    $("count").textContent = mine && v.i >= 0 && n > 1 ? (v.i + 1) + "/" + n : "";
  }
  $("toggle").addEventListener("click", () => {
    if (!v) return;
    const st = v.token === CFG.token ? v.state : "idle";
    if (st === "speaking" || st === "loading") v.pause();
    else if (st === "paused" || st === "blocked") v.resume();
    else v.start(job());
    render();
  });
  $("stop").addEventListener("click", () => { if (v && v.token === CFG.token) v.stop(); render(); });
  render();
  setInterval(render, 200);
</script></body></html>"""

# The brand mark: a saqya (ساقية), the water wheel that has lifted Nile water onto Egyptian fields for
# centuries — clay pots (qawadis) around a turning wheel. It turns slowly in the header (not with reduced motion).
WHEEL_SVG = """<svg class="aa-wheel-svg" viewBox="0 0 32 32" width="30" height="30" aria-hidden="true"><g class="aa-wheel"><g fill="none" stroke="#cbfffc" stroke-linecap="round"><circle cx="16" cy="13.6" r="8.6" stroke-width="1.5"/><path d="M18.23 14.50L23.97 16.82M16.94 15.81L19.36 21.52M15.10 15.83L12.78 21.57M13.79 14.54L8.08 16.96M13.77 12.70L8.03 10.38M15.06 11.39L12.64 5.68M16.90 11.37L19.22 5.63M18.21 12.66L23.92 10.24" stroke-width="1" opacity=".75"/><circle cx="16" cy="13.6" r="2.4" stroke-width="1"/></g><g fill="#cbfffc"><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(26.80 13.60)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(23.64 21.24)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(16.00 24.40)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(8.36 21.24)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(5.20 13.60)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(8.36 5.96)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(16.00 2.80)"/><path d="M-1.9 -1.1h3.8l-.7 2.5a1.2 1.2 0 0 1-1.2.9h0a1.2 1.2 0 0 1-1.2-.9z" transform="translate(23.64 5.96)"/></g></g><path d="M2.5 29.2c2.2-1.6 4.4-1.6 6.6 0s4.4 1.6 6.6 0 4.4-1.6 6.6 0 4.4 1.6 6.6 0" fill="none" stroke="#cbfffc" stroke-width="1.3" stroke-linecap="round" opacity=".8"/></svg>"""

BRAND_NAME = "SAQYA"
BRAND_NAME_AR = "ساقية"
