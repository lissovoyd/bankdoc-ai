// API Base URL
const API_BASE = '';

// State
let currentDoc = null;
let pdfDoc = null;
let currentPage = 1;
let statusPollTimer = null;
let lastSelectedDocId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});



async function openPdfAtPage(pageNum = 1) {
  if (!currentDoc) return;

  const pdfViewer = document.getElementById('pdfViewer');
  const pdfTitle = document.getElementById('pdfTitle');

  if (!pdfViewer || !pdfTitle) {
    console.error("PDF elements not found (pdfViewer/pdfTitle). Check HTML ids.");
    return;
  }

  pdfViewer.style.display = 'flex';

  try {
    const url = `${API_BASE}/uploads/${encodeURIComponent(currentDoc.filename)}`;

    // If another doc was opened before, reset the cached pdf
    if (!pdfDoc || pdfDoc._sourceUrl !== url) {
      pdfDoc = await pdfjsLib.getDocument(url).promise;
      // mark it so we know if doc changed (pdfjs doesn't expose url nicely)
      pdfDoc._sourceUrl = url;
    }

    currentPage = Math.max(1, Math.min(pageNum, pdfDoc.numPages));
    pdfTitle.textContent = `${currentDoc.title} - Page ${currentPage}`;
    await renderPage(currentPage);

  } catch (err) {
    console.error("openPdfAtPage error:", err);
    addMessage('system', `❌ PDF open failed: ${err.message || err}`);
  }
}

async function initializeApp() {
  loadDocuments();

  // Event listeners
  document.getElementById('uploadBtn').addEventListener('click', () => {
    document.getElementById('fileInput').click();
  });

  document.getElementById('fileInput').addEventListener('change', handleFileUpload);
  document.getElementById('askBtn').addEventListener('click', askQuestion);

  document.getElementById('questionInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  });

  // PDF controls
  document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
  document.getElementById('nextPage').addEventListener('click', () => changePage(1));
  document.getElementById('closePdf').addEventListener('click', closePdfViewer);

  // ✅ REMOVE auto-test behavior (it causes “blinking”)
  // Do not call window.testUpload() automatically.
}

// Load documents from API
async function loadDocuments() {
    const docList = document.getElementById('docList');
    docList.innerHTML = '<div class="loading">Loading...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/docs`);
        const docs = await response.json();
        
        if (docs.length === 0) {
            docList.innerHTML = '<div class="loading">No documents yet. Upload one to start!</div>';
            return;
        }
        
        docList.innerHTML = docs.map(doc => `
            <div class="doc-item" data-doc-id="${doc.id}" onclick="selectDocument(${doc.id})">
                <div class="doc-item-title">${doc.title}</div>
                <div class="doc-item-meta">
                    <span>${new Date(doc.uploaded_at).toLocaleDateString()}</span>
                    <span class="status-badge status-${doc.status.toLowerCase()}">${doc.status}</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading documents:', error);
        docList.innerHTML = '<div class="loading">Error loading documents</div>';
    }
}

// Handle file upload
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const progressSection = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    progressSection.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';
    
    try {
        const response = await fetch(`${API_BASE}/api/docs`, {
            method: 'POST',
            body: formData
        });
        
        const doc = await response.json();
        
        // Trigger extraction
        progressFill.style.width = '50%';
        progressText.textContent = 'Extracting text...';
        
        const extractResponse = await fetch(`${API_BASE}/api/docs/${doc.id}/extract`, {
            method: 'POST'
        });
        
        const extractResult = await extractResponse.json();
        
        // Poll for completion
        await pollTaskStatus(extractResult.task_id);
        
        progressFill.style.width = '100%';
        progressText.textContent = 'Complete!';
        
        setTimeout(() => {
            progressSection.style.display = 'none';
            loadDocuments();
            selectDocument(doc.id);
        }, 1000);
        
    } catch (error) {
        console.error('Upload error:', error);
        progressText.textContent = 'Upload failed';
    }
    
    // Reset input
    e.target.value = '';
}

// Poll task status
async function pollTaskStatus(taskId) {
    while (true) {

        
        const response = await fetch(`${API_BASE}/tasks/${taskId}`);
        const task = await response.json();
        
        if (task.state === 'SUCCESS') {
            return task.result;
        } else if (task.state === 'FAILURE') {
            throw new Error(task.error);
        }
        
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}


// Select document - FIXED VERSION
async function selectDocument(docId) {
  console.log('Selecting document:', docId);

  // stop previous polling (prevents blinking)
  if (statusPollTimer) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }

  lastSelectedDocId = docId;

  // Update UI active state
  document.querySelectorAll('.doc-item').forEach(item => item.classList.remove('active'));
  const selectedItem = document.querySelector(`[data-doc-id="${docId}"]`);
  if (selectedItem) selectedItem.classList.add('active');

  try {
    const response = await fetch(`${API_BASE}/api/docs/${docId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    currentDoc = await response.json();
    console.log('Document loaded:', currentDoc);

    // Update header - REMOVE document title from chat
    document.getElementById('chatTitle').textContent = `💬 Ask Questions`;  // ← Changed
    document.getElementById('chatSubtitle').textContent =
      `Status: ${currentDoc.status} | Uploaded: ${new Date(currentDoc.uploaded_at).toLocaleDateString()}`;

    // Clear messages
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';

    const chatInputContainer = document.getElementById('chatInputContainer');
    const pdfViewer = document.getElementById('pdfViewer');

    // ✅ Show PDF when EXTRACTED or EMBEDDED
    const canShowPdf = (currentDoc.status === 'EXTRACTED' || currentDoc.status === 'EMBEDDED');
    pdfViewer.style.display = canShowPdf ? 'flex' : 'none';

    // ✅ Only allow questions when EMBEDDED
    const canAsk = (currentDoc.status === 'EMBEDDED');
    chatInputContainer.style.display = canAsk ? 'flex' : 'none';

    if (canShowPdf) {
      addMessage('system', `📄 PDF ready. ${canAsk ? "✅ You can ask questions now." : "⏳ Embedding is still running (questions will unlock when done)."}`
      );
      await openPdfAtPage(1);
    } else {
      addMessage('system', `⏳ Document is ${currentDoc.status}. Waiting for extraction...`);
      closePdfViewer();
    }

    // ✅ Controlled polling only if not embedded yet
    if (!canAsk) {
      statusPollTimer = setInterval(async () => {
        // only poll if user is still on the same doc
        if (lastSelectedDocId !== docId) return;

        try {
          const r = await fetch(`${API_BASE}/api/docs/${docId}`);
          if (!r.ok) return;

          const updated = await r.json();

          // if status changed, refresh the UI once
          if (updated.status !== currentDoc.status) {
            currentDoc = updated;
            console.log("Status changed -> reloading selected doc UI:", updated.status);
            await selectDocument(docId);
            loadDocuments();
          }
        } catch (_) {
          // ignore polling errors
        }
      }, 2500);
    }

  } catch (error) {
    console.error('Error loading document:', error);
    addMessage('system', `❌ Error loading document: ${error.message}`);
  }
}

async function jumpToSource(src) {
  if (!currentDoc) return;

  await openPdfAtPage(src.page_num);

  // small wait so textLayer exists
  await new Promise(r => setTimeout(r, 120));

  // highlight exact chunk using excerpt
  if (src.text_excerpt) {
    highlightChunkByExcerpt(src.text_excerpt);
  }
}


// Ask question
async function askQuestion() {
    if (!currentDoc) return;
    
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    // Add question to chat
    addMessage('user', question);
    input.value = '';
    
    // Show loading
    const loadingId = addMessage('assistant', '...', true);
    
    try {
        const response = await fetch(`${API_BASE}/api/docs/${currentDoc.id}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const result = await response.json();
        
        // Remove loading
        document.getElementById(loadingId)?.remove();
        
        // Add answer
        addMessage('assistant', result.answer, false, result.sources);
        
    } catch (error) {
        console.error('Error asking question:', error);
        document.getElementById(loadingId)?.remove();
        addMessage('system', 'Error: Could not get answer');
    }
}

// Add message to chat
function addMessage(type, text, isLoading = false, sources = null) {
    const messages = document.getElementById('chatMessages');
    const id = `msg-${Date.now()}`;
    
    let html = '';
    
    if (type === 'user') {
        html = `<div class="message" id="${id}"><div class="message-question">${escapeHtml(text)}</div></div>`;
    } else if (type === 'assistant') {
        html = `
            <div class="message" id="${id}">
                <div class="message-answer">
                    <div class="answer-text">${escapeHtml(text)}</div>
                    ${sources ? renderSources(sources) : ''}
                </div>
            </div>
        `;
    } else {
        html = `<div class="message" id="${id}"><div class="message-answer">${escapeHtml(text)}</div></div>`;
    }
    
    messages.insertAdjacentHTML('beforeend', html);
    messages.scrollTop = messages.scrollHeight;
    
    return id;
}

// Render sources
// Render sources with click handler
function renderSources(sources) {
    return `
        <div class="answer-sources">
            <h4>📎 Sources:</h4>
            ${sources.map((src, idx) => {
                const srcJson = escapeHtml(JSON.stringify(src));
                return `
                    <span class="source-ref" onclick='jumpToSource(${srcJson})'>
                        [${idx + 1}] Page ${src.page_num}
                    </span>
                `;
            }).join('')}
        </div>
    `;
}

// Jump to page in PDF
async function jumpToPage(pageNum) {
  if (!currentDoc) return;

  const pdfViewer = document.getElementById('pdfViewer');
  const pdfTitle = document.getElementById('pdfTitle');

  if (!pdfViewer || !pdfTitle) {
    console.error("pdfViewer/pdfTitle not found");
    return;
  }

  pdfViewer.style.display = 'flex';

  try {
    if (!pdfDoc) {
      const url = `${API_BASE}/uploads/${encodeURIComponent(currentDoc.filename)}`;
      pdfDoc = await pdfjsLib.getDocument(url).promise;
    }

    currentPage = Math.max(1, Math.min(pageNum, pdfDoc.numPages));
    pdfTitle.textContent = `${currentDoc.title} - Page ${currentPage}`;

    await renderPage(currentPage);

  } catch (error) {
    console.error('Error loading PDF:', error);
    addMessage('system', `❌ PDF load failed: ${error.message || error}`);
  }
}

// Render PDF page with TEXT LAYER (enables copy/paste)
let lastHighlightedSpans = [];

async function renderPage(pageNum) {
  const page = await pdfDoc.getPage(pageNum);

  const wrap = document.getElementById('pdfWrap');
  const pageContainer = document.getElementById('pdfPageContainer');
  const canvas = document.getElementById('pdfCanvas');
  const ctx = canvas.getContext('2d');

  // pick scale based on available width
  const maxWidth = wrap.clientWidth - 28; // wrap padding-ish
  const viewport1 = page.getViewport({ scale: 1 });
  const scale = Math.max(1, maxWidth / viewport1.width);
  const viewport = page.getViewport({ scale });

  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);

  // render canvas
  await page.render({ canvasContext: ctx, viewport }).promise;

  // remove old text layer
  const oldTextLayer = pageContainer.querySelector('.textLayer');
  if (oldTextLayer) oldTextLayer.remove();

  // clear old chunk highlights
  clearChunkHighlights();

  // create new text layer
  const textLayerDiv = document.createElement('div');
  textLayerDiv.className = 'textLayer';
  pageContainer.appendChild(textLayerDiv);

  // render text layer (this enables copy/paste)
  const textContent = await page.getTextContent();

  const textDivs = [];
  const renderTask = pdfjsLib.renderTextLayer({
    textContentSource: textContent,
    container: textLayerDiv,
    viewport,
    textDivs
  });

  // pdf.js 3.x returns an object with a promise
  if (renderTask?.promise) await renderTask.promise;

  document.getElementById('pageInfo').textContent =
    `Page ${pageNum} of ${pdfDoc.numPages}`;

  // stash text info for chunk highlighting (used in #3)
  textLayerDiv._textItems = textContent.items;
  textLayerDiv._textDivs = textDivs;
}


// Change page
function changePage(delta) {
  if (!pdfDoc) return;

  const newPage = currentPage + delta;
  if (newPage < 1 || newPage > pdfDoc.numPages) return;

  currentPage = newPage;

  const hl = document.getElementById("pdfHighlight");
  if (hl) hl.style.display = "none";

  renderPage(currentPage);
}

// Close PDF viewer
function closePdfViewer() {
  const pdfViewer = document.getElementById('pdfViewer');
  if (pdfViewer) pdfViewer.style.display = 'none';

  pdfDoc = null;

  const canvas = document.getElementById('pdfCanvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}


// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


function highlightChunkOnPage(src) {
  const hl = document.getElementById("pdfHighlight");
  const canvas = document.getElementById("pdfCanvas");
  if (!hl || !canvas) return;

  // if metadata missing, just do a quick flash at top
  const start = (typeof src.start_sentence === "number") ? src.start_sentence : null;
  const end = (typeof src.end_sentence === "number") ? src.end_sentence : null;

  const canvasH = canvas.height;

  // Default highlight size
  let topPx = Math.floor(canvasH * 0.12);
  let heightPx = Math.floor(canvasH * 0.18);

  // If we have sentence range, map it to page height (approx)
  // We assume ~50 sentences per page (rough), but we can scale by end_sentence.
  if (start !== null && end !== null && end >= start) {
    const approxTotal = Math.max(end + 10, 60); // avoid tiny totals
    const startRatio = start / approxTotal;
    const endRatio = end / approxTotal;

    topPx = Math.floor(canvasH * startRatio);
    heightPx = Math.max(40, Math.floor(canvasH * (endRatio - startRatio)));
  }

  // Clamp
  topPx = Math.max(10, Math.min(topPx, canvasH - 60));
  heightPx = Math.max(35, Math.min(heightPx, canvasH - topPx - 10));

  hl.style.top = `${topPx}px`;
  hl.style.height = `${heightPx}px`;
  hl.style.display = "block";
  hl.style.opacity = "1";

  // fade out after a bit (optional)
  clearTimeout(hl._hideTimer);
  hl._hideTimer = setTimeout(() => {
    hl.style.opacity = "0";
    setTimeout(() => (hl.style.display = "none"), 200);
  }, 2200);
}

function clearChunkHighlights() {
  // remove span highlights
  for (const sp of lastHighlightedSpans) {
    sp.classList.remove('pdf-chunk-highlight');
  }
  lastHighlightedSpans = [];
}

function normalizeForMatch(s) {
  return (s || "")
    .replace(/\s+/g, " ")
    .replace(/[“”«»]/g, '"')
    .trim()
    .toLowerCase();
}

function highlightChunkByExcerpt(excerpt) {
  const pageContainer = document.getElementById('pdfPageContainer');
  const textLayer = pageContainer.querySelector('.textLayer');
  if (!textLayer || !textLayer._textItems || !textLayer._textDivs) return;

  clearChunkHighlights();

  const items = textLayer._textItems;   // pdf.js text items
  const divs  = textLayer._textDivs;    // span divs for each item

  // Build a full page string with mapping back to item indices
  let full = "";
  const map = []; // map char index -> item index (by ranges)
  for (let i = 0; i < items.length; i++) {
    const t = items[i].str || "";
    const start = full.length;
    full += t + " ";
    const end = full.length;
    map.push({ i, start, end });
  }

  const hay = normalizeForMatch(full);
  const needle = normalizeForMatch(excerpt).slice(0, 160); // match only first part
  if (!needle || needle.length < 12) return;

  const pos = hay.indexOf(needle);
  if (pos === -1) {
    // fallback: try a shorter needle
    const shortNeedle = needle.slice(0, 80);
    if (shortNeedle.length < 12) return;
    const pos2 = hay.indexOf(shortNeedle);
    if (pos2 === -1) return;
    return highlightRangeByCharPos(map, divs, pos2, pos2 + shortNeedle.length);
  }

  highlightRangeByCharPos(map, divs, pos, pos + needle.length);
}

function highlightRangeByCharPos(map, divs, startChar, endChar) {
  // find which items overlap the char range
  const hits = [];
  for (const r of map) {
    if (r.end < startChar) continue;
    if (r.start > endChar) break;
    hits.push(r.i);
  }

  // highlight spans
  for (const idx of hits) {
    const sp = divs[idx];
    if (sp) {
      sp.classList.add('pdf-chunk-highlight');
      lastHighlightedSpans.push(sp);
    }
  }

  // scroll to first highlighted span
  const first = divs[hits[0]];
  if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });

  // auto fade after a bit (optional)
  setTimeout(() => clearChunkHighlights(), 2600);
}
