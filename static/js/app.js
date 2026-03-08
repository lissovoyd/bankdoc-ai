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

  document.getElementById('sidebarToggle').addEventListener('click', () => {
    document.querySelector('.container').classList.toggle('sidebar-collapsed');
  });

  // Drag-to-resize between PDF and Chat panels
  const resizeHandle = document.getElementById('panelResizeHandle');
  const mainContent = document.querySelector('.main-content.three-col');

  resizeHandle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const cols = getComputedStyle(mainContent).gridTemplateColumns.split(' ');
    const startPdfPx = parseFloat(cols[0]);
    resizeHandle.classList.add('dragging');

    function onMove(e) {
      const delta = e.clientX - startX;
      const totalWidth = mainContent.clientWidth;
      const newPdf = Math.max(300, Math.min(totalWidth - 270, startPdfPx + delta));
      const newChat = Math.max(220, totalWidth - newPdf - 13);
      mainContent.style.gridTemplateColumns = `${newPdf}px 5px ${newChat}px`;
    }
    function onUp() {
      resizeHandle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

// Delete document
async function deleteDocument(docId, title) {
    if (!confirm(`Delete "${title}"?\n\nThis will remove the PDF, all extracted text, and embeddings.`)) return;

    try {
        const response = await fetch(`${API_BASE}/api/docs/${docId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        // If the deleted doc was active, clear the view
        if (currentDoc && currentDoc.id === docId) {
            currentDoc = null;
            closePdfViewer();
            document.getElementById('chatMessages').innerHTML = '<div class="welcome-message"><h3>Welcome to BankDoc AI</h3><p>Select a document to begin asking questions.</p></div>';
            document.getElementById('chatInputContainer').style.display = 'none';
            document.getElementById('chatSubtitle').textContent = 'Select a document to start';
        }

        loadDocuments();
    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete document: ' + error.message);
    }
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
                <button class="doc-item-delete" onclick="event.stopPropagation(); deleteDocument(${doc.id}, '${escapeHtml(doc.title)}')" title="Delete document">&#x2715;</button>
                <div class="doc-item-title">${escapeHtml(doc.title)}</div>
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

  // wait for text layer to finish rendering
  await new Promise(r => setTimeout(r, 350));

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
        
        // Add answer with confidence info
        addMessage('assistant', result.answer, false, result.sources, {
            confidence: result.confidence,
            grounded: result.grounded,
            declined: result.declined,
        });
        
    } catch (error) {
        console.error('Error asking question:', error);
        document.getElementById(loadingId)?.remove();
        addMessage('system', 'Error: Could not get answer');
    }
}

// Add message to chat
function addMessage(type, text, isLoading = false, sources = null, meta = null) {
    const messages = document.getElementById('chatMessages');
    const id = `msg-${Date.now()}`;

    let html = '';

    if (type === 'user') {
        html = `<div class="message" id="${id}"><div class="message-question">${escapeHtml(text)}</div></div>`;
    } else if (type === 'assistant') {
        const confidenceBadge = meta ? renderConfidenceBadge(meta) : '';
        html = `
            <div class="message" id="${id}">
                <div class="message-answer ${meta && meta.declined ? 'answer-declined' : ''}">
                    ${confidenceBadge}
                    <div class="answer-text">${escapeHtml(text)}</div>
                    ${sources && sources.length ? renderSources(sources) : ''}
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

// Render confidence badge
function renderConfidenceBadge(meta) {
    if (!meta || typeof meta.confidence !== 'number') return '';
    const score = meta.confidence;
    let level, label;
    if (meta.declined) {
        level = 'red'; label = 'Нет данных';
    } else if (score >= 70) {
        level = 'green'; label = `Высокая ${score.toFixed(0)}%`;
    } else if (score >= 45) {
        level = 'yellow'; label = `Средняя ${score.toFixed(0)}%`;
    } else {
        level = 'red'; label = `Низкая ${score.toFixed(0)}%`;
    }
    return `<span class="confidence-badge confidence-${level}" title="Релевантность ответа: ${score.toFixed(1)}%">${label}</span>`;
}

// Render sources with click handler — full-width cards with excerpt
function renderSources(sources) {
    return `
        <div class="answer-sources">
            <h4>Sources:</h4>
            ${sources.slice(0, 5).map((src, idx) => {
                const srcJson = escapeHtml(JSON.stringify(src));
                const score = src.relevance_score || 0;
                const scoreClass = score >= 70 ? 'source-score-high' : score >= 45 ? 'source-score-med' : 'source-score-low';
                const sectionLabel = src.section_title ? `<span class="source-ref-section">${escapeHtml(src.section_title)}</span>` : '';
                const excerpt = src.text_excerpt ? src.text_excerpt.replace(/\s+/g, ' ').trim().slice(0, 140) : '';
                return `
                    <div class="source-ref" onclick='jumpToSource(${srcJson})'>
                        <div class="source-ref-header">
                            <span class="source-ref-page">[${idx + 1}] Страница ${src.page_num}</span>
                            <span class="source-ref-score ${scoreClass}">${score.toFixed(0)}%</span>
                        </div>
                        ${sectionLabel}
                        ${excerpt ? `<div class="source-ref-excerpt">${escapeHtml(excerpt)}…</div>` : ''}
                    </div>
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

  // pick scale based on available width, zoom out slightly
  const maxWidth = wrap.clientWidth - 28;
  const viewport1 = page.getViewport({ scale: 1 });
  const scale = Math.max(0.5, (maxWidth / viewport1.width) * 0.82);
  const viewport = page.getViewport({ scale });

  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);

  // render canvas
  await page.render({ canvasContext: ctx, viewport }).promise;

  // remove old text layer
  const oldTextLayer = pageContainer.querySelector('.textLayer');
  if (oldTextLayer) oldTextLayer.remove();

  // clear old chunk highlights and overlay
  clearChunkHighlights();
  clearPageOverlay();

  // create new text layer — sized via CSS inset:0 (fills page container)
  const textLayerDiv = document.createElement('div');
  textLayerDiv.className = 'textLayer';
  // pdfjs 3.11.174 requires --scale-factor for correct span positioning
  textLayerDiv.style.setProperty('--scale-factor', viewport.scale);
  pageContainer.appendChild(textLayerDiv);

  // pdfjs 3.x text layer rendering (textContentSource = new API)
  const textDivs = [];
  const textContent = await page.getTextContent();
  const renderTask = pdfjsLib.renderTextLayer({
    textContentSource: textContent,
    container: textLayerDiv,
    viewport,
    textDivs,
  });
  if (renderTask?.promise) await renderTask.promise;

  console.log('[PDF] textLayer spans created:', textDivs.length);

  document.getElementById('pageInfo').textContent =
    `Page ${pageNum} of ${pdfDoc.numPages}`;

  // Store for chunk highlighting
  textLayerDiv._textItems = textContent.items;
  textLayerDiv._textDivs = textDivs;
}


// Change page
function changePage(delta) {
  if (!pdfDoc) return;

  const newPage = currentPage + delta;
  if (newPage < 1 || newPage > pdfDoc.numPages) return;

  currentPage = newPage;

  clearPageOverlay();
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


function showPageOverlay() {
  // Fallback: show a full-page tint when text match fails
  const hl = document.getElementById("pdfHighlightLayer");
  if (!hl) return;
  hl.style.background = "rgba(37, 99, 235, 0.07)";
  hl.style.outline = "2px solid rgba(37, 99, 235, 0.3)";
  hl.style.borderRadius = "10px";
}

function clearPageOverlay() {
  const hl = document.getElementById("pdfHighlightLayer");
  if (!hl) return;
  hl.style.background = "";
  hl.style.outline = "";
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
  if (!textLayer || !textLayer._textItems || !textLayer._textDivs) {
    showPageOverlay();
    return;
  }

  clearChunkHighlights();
  clearPageOverlay();

  const items = textLayer._textItems;
  const divs  = textLayer._textDivs;

  // Build a full page string with mapping back to item indices
  let full = "";
  const map = [];
  for (let i = 0; i < items.length; i++) {
    const t = items[i].str || "";
    const start = full.length;
    full += t + " ";
    map.push({ i, start, end: full.length });
  }

  const hay = normalizeForMatch(full);
  const baseNeedle = normalizeForMatch(excerpt);

  // Strategy: find the chunk's position using a short needle, then highlight
  // the full excerpt length from that position (not just the needle).
  const offsets = [0, 80, 150];
  const lengths = [200, 120, 70, 40];

  for (const off of offsets) {
    if (off >= baseNeedle.length) continue;
    const sub = baseNeedle.slice(off);
    for (const len of lengths) {
      const needle = sub.slice(0, len);
      if (needle.length < 15) break;
      const pos = hay.indexOf(needle);
      if (pos !== -1) {
        // Extend highlight to cover the full excerpt, anchored at the match
        const hlStart = Math.max(0, pos - off);
        const hlEnd = Math.min(hay.length, hlStart + baseNeedle.length);
        highlightRangeByCharPos(map, divs, hlStart, hlEnd);
        return;
      }
    }
  }

  // All needle lengths failed — show page overlay as fallback
  showPageOverlay();
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
  // highlights persist until page changes
}
