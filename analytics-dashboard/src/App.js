import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

// Design tokens 
const C = {
  bg:          '#080c14',
  surface:     '#0e1420',
  card:        '#131929',
  cardBorder:  '#1e2d45',
  hover:       '#1a2538',
  text:        '#e8edf5',
  muted:       '#7a8fa8',
  dim:         '#3d5068',
  primary:     '#4f8ef7',
  primaryDim:  'rgba(79,142,247,0.15)',
  green:       '#30d988',
  greenDim:    'rgba(48,217,136,0.15)',
  purple:      '#a67cf5',
  purpleDim:   'rgba(166,124,245,0.15)',
  amber:       '#f0b840',
  amberDim:    'rgba(240,184,64,0.15)',
  red:         '#f0534a',
  teal:        '#2dcdc8',
};

const API   = 'http://localhost:8000';
const MJPEG = 'http://localhost:8001/video_feed';

//Inject global styles 
const GlobalStyle = () => (
  <style>{`
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: ${C.bg}; color: ${C.text}; font-family: 'Inter', system-ui, sans-serif; }
    @keyframes pulse-ring {
      0%   { box-shadow: 0 0 0 0 ${C.greenDim}; }
      70%  { box-shadow: 0 0 0 6px transparent; }
      100% { box-shadow: 0 0 0 0 transparent; }
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: ${C.surface}; }
    ::-webkit-scrollbar-thumb { background: ${C.dim}; border-radius: 3px; }
  `}</style>
);

// Shared atoms 
const card = (extra = {}) => ({
  background:   C.card,
  border:       `1px solid ${C.cardBorder}`,
  borderRadius: 14,
  padding:      '1.4rem 1.6rem',
  ...extra,
});

const LiveBadge = () => (
  <span style={{ display:'inline-flex', alignItems:'center', gap:6,
    background: C.greenDim, border:`1px solid ${C.green}`,
    borderRadius:20, padding:'3px 10px' }}>
    <span style={{ width:7, height:7, borderRadius:'50%', background:C.green,
      animation:'pulse-ring 1.8s ease-out infinite' }} />
    <span style={{ fontSize:11, fontWeight:700, color:C.green, letterSpacing:1 }}>LIVE</span>
  </span>
);

const Spinner = ({ size = 18 }) => (
  <span style={{ display:'inline-block', width:size, height:size,
    border:`2px solid ${C.dim}`, borderTopColor:C.primary,
    borderRadius:'50%', animation:'spin 0.7s linear infinite' }} />
);

const SectionCard = ({ title, children, style = {} }) => (
  <div style={{ ...card(), ...style, animation:'fadeUp 0.3s ease' }}>
    {title && <p style={{ fontSize:12, fontWeight:700, color:C.muted,
      textTransform:'uppercase', letterSpacing:1.5, marginBottom:'1.2rem' }}>{title}</p>}
    {children}
  </div>
);

const StatCard = ({ label, value, sub, accent = C.primary, icon }) => (
  <div style={{ ...card(), borderTop:`3px solid ${accent}`, animation:'fadeUp 0.3s ease' }}>
    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
      <span style={{ fontSize:11, fontWeight:700, color:C.muted,
        textTransform:'uppercase', letterSpacing:1.5 }}>{label}</span>
      {icon && <span style={{ fontSize:20 }}>{icon}</span>}
    </div>
    <div style={{ fontSize:'2.4rem', fontWeight:800, color:accent,
      lineHeight:1, margin:'0.6rem 0 0.3rem', letterSpacing:-1 }}>{value}</div>
    {sub && <div style={{ fontSize:11, color:C.dim }}>{sub}</div>}
  </div>
);

const tooltipStyle = {
  contentStyle: { background:C.surface, border:`1px solid ${C.cardBorder}`,
    borderRadius:10, color:C.text, fontSize:12 },
  labelStyle: { color:C.muted, marginBottom:4 },
};

//  Tab navigation 
const TABS = [
  { id:'analytics', label:'Analytics',       icon:'📊' },
  { id:'live',      label:'Live Feed',        icon:'📹' },
  { id:'upload',    label:'Upload Video',     icon:'⬆' },
  { id:'output',    label:'Processed Output', icon:'🎬' },
];

const TabBar = ({ active, onChange }) => (
  <div style={{ display:'flex', borderBottom:`1px solid ${C.cardBorder}`,
    marginBottom:'1.8rem', gap:4 }}>
    {TABS.map(t => (
      <button key={t.id} onClick={() => onChange(t.id)} style={{
        background:'none', border:'none', cursor:'pointer', outline:'none',
        padding:'0.75rem 1.25rem', display:'flex', alignItems:'center', gap:8,
        fontSize:'0.875rem', fontWeight: active===t.id ? 700 : 500,
        color: active===t.id ? C.primary : C.muted,
        borderBottom: active===t.id ? `2px solid ${C.primary}` : '2px solid transparent',
        marginBottom:-1, transition:'all 0.15s', whiteSpace:'nowrap',
      }}>
        <span>{t.icon}</span> {t.label}
      </button>
    ))}
  </div>
);

// Analytics Tab 
const AnalyticsTab = ({ latest, history, summary, hourly, embeddings }) => {
  const peakQueue  = summary.peak_queue ?? Math.max(0, ...history.map(d => d.queue_length));
  const avgQueue   = summary.avg_queue  ?? 0;

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.4rem', animation:'fadeUp 0.3s ease' }}>

      {/* KPI row */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(190px,1fr))', gap:'1rem' }}>
        <StatCard label="Live Queue"       value={latest.queue_length ?? 0}     sub="People on camera now"        accent={C.primary} icon="👥" />
        <StatCard label="Session Traffic"  value={latest.foot_traffic ?? 0}     sub="Unique people this session"  accent={C.green}   icon="🚶" />
        <StatCard label="All-Time Unique"  value={latest.unique_visitors ?? 0}  sub="Cross-session identities"   accent={C.purple}  icon="🔑" />
        <StatCard label="Peak Queue Today" value={peakQueue}                    sub="Highest simultaneous count" accent={C.amber}   icon="📈" />
        <StatCard label="Avg Queue"        value={avgQueue}                     sub="24 h rolling average"       accent={C.teal}    icon="⌀"  />
        <StatCard label="Privacy Status"   value="100%"                         sub="All faces obfuscated"       accent={C.green}   icon="🛡" />
      </div>

      {/* Traffic over time */}
      <SectionCard title="Traffic Over Time">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={history} margin={{ top:5, right:20, bottom:5, left:0 }}>
            <defs>
              <linearGradient id="gQueue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={C.primary} stopOpacity={0.25} />
                <stop offset="95%" stopColor={C.primary} stopOpacity={0}    />
              </linearGradient>
              <linearGradient id="gUnique" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={C.purple} stopOpacity={0.25} />
                <stop offset="95%" stopColor={C.purple} stopOpacity={0}    />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={C.cardBorder} vertical={false} />
            <XAxis dataKey="time" stroke={C.dim} fontSize={11} tickMargin={8} />
            <YAxis stroke={C.dim} fontSize={11} allowDecimals={false} />
            <Tooltip {...tooltipStyle} />
            <Legend wrapperStyle={{ fontSize:12, color:C.muted, paddingTop:12 }} />
            <Area type="monotone" dataKey="queue_length"   name="Live Queue"       stroke={C.primary} fill="url(#gQueue)"  strokeWidth={2} dot={false} />
            <Area type="monotone" dataKey="unique_visitors" name="Unique Visitors" stroke={C.purple}  fill="url(#gUnique)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </SectionCard>

      {/* Hourly distribution */}
      <SectionCard title="Hourly Distribution (last 24 h)">
        {hourly.length === 0
          ? <p style={{ color:C.dim, fontSize:13, textAlign:'center', padding:'2rem 0' }}>No hourly data yet — keep the system running.</p>
          : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hourly} margin={{ top:5, right:20, bottom:5, left:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.cardBorder} vertical={false} />
              <XAxis dataKey="hour" stroke={C.dim} fontSize={11} tickMargin={8} />
              <YAxis stroke={C.dim} fontSize={11} allowDecimals={false} />
              <Tooltip {...tooltipStyle} />
              <Legend wrapperStyle={{ fontSize:12, color:C.muted, paddingTop:12 }} />
              <Bar dataKey="peak_queue" name="Peak Queue" fill={C.primary} radius={[4,4,0,0]} />
              <Bar dataKey="avg_queue"  name="Avg Queue"  fill={C.teal}    radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </SectionCard>

      {/* Identities table */}
      <SectionCard title={`Stored Obfuscated Identities — ${embeddings.length} total`}>
        <div style={{ overflowX:'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
            <thead>
              <tr style={{ borderBottom:`1px solid ${C.cardBorder}` }}>
                {['Person ID','First Seen','Embedding Preview','Dimensions'].map(h => (
                  <th key={h} style={{ padding:'0.6rem 0.8rem', textAlign:'left',
                    color:C.muted, fontWeight:600, fontSize:11,
                    textTransform:'uppercase', letterSpacing:1 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {embeddings.length === 0
                ? <tr><td colSpan={4} style={{ textAlign:'center', color:C.dim,
                    padding:'2rem', fontSize:13 }}>No identities captured yet.</td></tr>
                : embeddings.slice(0, 8).map((p, i) => (
                  <tr key={i} style={{ borderBottom:`1px solid ${C.cardBorder}`,
                    transition:'background 0.1s' }}
                    onMouseEnter={e => e.currentTarget.style.background = C.hover}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ padding:'0.7rem 0.8rem' }}>
                      <span style={{ background:C.primaryDim, color:C.primary,
                        borderRadius:6, padding:'3px 10px', fontWeight:700, fontSize:12 }}>
                        #{p.person_id}
                      </span>
                    </td>
                    <td style={{ padding:'0.7rem 0.8rem', color:C.muted, fontSize:12 }}>{p.timestamp}</td>
                    <td style={{ padding:'0.7rem 0.8rem', color:C.dim,
                      fontFamily:'monospace', fontSize:11 }}>
                      [{p.embedding.slice(0,4).map(n => n.toFixed(3)).join(', ')} …]
                    </td>
                    <td style={{ padding:'0.7rem 0.8rem' }}>
                      <span style={{ background:C.purpleDim, color:C.purple,
                        borderRadius:6, padding:'2px 8px', fontSize:11 }}>
                        {p.embedding.length}d
                      </span>
                    </td>
                  </tr>
              ))}
            </tbody>
          </table>
          {embeddings.length > 8 && (
            <p style={{ textAlign:'center', color:C.dim, fontSize:12, marginTop:'1rem' }}>
              Showing 8 of {embeddings.length} identities
            </p>
          )}
        </div>
      </SectionCard>
    </div>
  );
};

// Live Feed Tab 
const LiveTab = () => {
  const [error, setError] = useState(false);
  return (
    <div style={{ animation:'fadeUp 0.3s ease' }}>

      {/* title row */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center',
        marginBottom:'0.75rem', padding:'0 2rem' }}>
        <div>
          <h2 style={{ fontSize:'1.1rem', fontWeight:700, color:C.text }}>Live Obfuscated Feed</h2>
        </div>
        {!error && <LiveBadge />}
      </div>

      {/* Full-bleed video — breaks out of the parent's side padding */}
      <div style={{
        marginLeft:  '-2rem',
        marginRight: '-2rem',
        background:  '#000',
        height:      'calc(100vh - 160px)',
        overflow:    'hidden',
      }}>
        {error ? (
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
            justifyContent:'center', height:'100%', color:C.muted }}>
            <p style={{ fontSize:'2rem', marginBottom:'1rem' }}>📷</p>
            <p style={{ fontWeight:600, color:C.text, marginBottom:8 }}>Stream unavailable</p>
            <p style={{ fontSize:13 }}>Make sure <code style={{ color:C.primary }}>app.py</code> is running on port 8001.</p>
          </div>
        ) : (
          <img
            src={MJPEG}
            alt="Live feed"
            style={{ width:'100%', height:'100%', objectFit:'contain', display:'block' }}
            onError={() => setError(true)}
          />
        )}
      </div>

    </div>
  );
};

//  Upload Tab 
const UploadTab = ({ onProcessingDone }) => {
  const [file,         setFile]         = useState(null);
  const [dragging,     setDragging]     = useState(false);
  const [uploading,    setUploading]    = useState(false);
  const [statusData,   setStatusData]   = useState(null);
  const [pollTimer,    setPollTimer]    = useState(null);
  const inputRef = useRef();

  const stopPolling = useCallback(() => {
    setPollTimer(t => { if (t) clearInterval(t); return null; });
  }, []);

  const startPolling = useCallback(() => {
    const t = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/api/video/status`);
        const data = await res.json();
        setStatusData(data);
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(t);
          if (data.status === 'done') onProcessingDone();
        }
      } catch { /* network hiccup — keep polling */ }
    }, 1000);
    setPollTimer(t);
  }, [onProcessingDone]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setStatusData({ status: 'uploading' });
    stopPolling();
    try {
      const body = new FormData();
      body.append('file', file);
      await fetch(`${API}/api/video/upload`, { method:'POST', body });
      setUploading(false);
      setStatusData({ status:'processing', progress:0 });
      startPolling();
    } catch (err) {
      setUploading(false);
      setStatusData({ status:'error', error: err.message });
    }
  };

  const status = statusData?.status;
  const progress = statusData?.progress ?? 0;
  const isIdle = !status || status === 'idle';
  const isProcessing = status === 'uploading' || status === 'processing';

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.4rem', animation:'fadeUp 0.3s ease' }}>
      <div>
        <h2 style={{ fontSize:'1.1rem', fontWeight:700, color:C.text }}>Upload Video for Processing</h2>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => !isProcessing && inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragging ? C.primary : file ? C.green : C.cardBorder}`,
          borderRadius: 14, padding:'3rem 2rem', textAlign:'center',
          cursor: isProcessing ? 'default' : 'pointer',
          background: dragging ? C.primaryDim : file ? C.greenDim : C.card,
          transition:'all 0.2s',
        }}>
        <input ref={inputRef} type="file" accept="video/*" style={{ display:'none' }}
          onChange={e => e.target.files[0] && setFile(e.target.files[0])} />
        {file ? (
          <>
            <p style={{ fontSize:'2rem', marginBottom:8 }}>🎥</p>
            <p style={{ fontWeight:700, color:C.green, marginBottom:4 }}>{file.name}</p>
            <p style={{ fontSize:12, color:C.muted }}>
              {(file.size / 1024 / 1024).toFixed(1)} MB — click to change
            </p>
          </>
        ) : (
          <>
            <p style={{ fontSize:'2.5rem', marginBottom:8 }}>⬆</p>
            <p style={{ fontWeight:600, color:C.text, marginBottom:4 }}>Drop a video file here</p>
            <p style={{ fontSize:12, color:C.muted }}>or click to browse — MP4, MOV, AVI</p>
          </>
        )}
      </div>

      {/* Upload button */}
      {!isProcessing && (
        <button onClick={handleUpload} disabled={!file}
          style={{
            padding:'0.8rem 2rem', borderRadius:10, border:'none',
            background: file ? C.primary : C.dim,
            color:'#fff', fontWeight:700, fontSize:14,
            cursor: file ? 'pointer' : 'not-allowed',
            transition:'opacity 0.15s', alignSelf:'flex-start',
          }}>
          {uploading ? 'Uploading…' : 'Process Video'}
        </button>
      )}

      {/* Progress */}
      {isProcessing && (
        <SectionCard>
          <div style={{ display:'flex', alignItems:'center', gap:12, marginBottom:'1rem' }}>
            <Spinner />
            <span style={{ color:C.text, fontWeight:600, fontSize:14 }}>
              {status === 'uploading' ? 'Uploading video…' : `Processing… ${progress.toFixed(1)}%`}
            </span>
          </div>
          <div style={{ background:C.surface, borderRadius:8, height:8, overflow:'hidden' }}>
            <div style={{
              height:'100%', borderRadius:8,
              background:`linear-gradient(90deg, ${C.primary}, ${C.purple})`,
              width:`${status === 'uploading' ? 5 : progress}%`,
              transition:'width 0.5s ease',
            }} />
          </div>
          {statusData?.total_frames > 0 && (
            <p style={{ fontSize:12, color:C.muted, marginTop:8 }}>
              {statusData.processed_frames} / {statusData.total_frames} frames
            </p>
          )}
        </SectionCard>
      )}

      {/* Done */}
      {status === 'done' && (
        <div style={{ ...card(), borderColor:C.green, background:C.greenDim }}>
          <p style={{ color:C.green, fontWeight:700, fontSize:14 }}>
            ✓ Processing complete — switch to the Processed Output tab to watch the result.
          </p>
        </div>
      )}

      {/* Error */}
      {status === 'error' && (
        <div style={{ ...card(), borderColor:C.red, background:'rgba(240,83,74,0.1)' }}>
          <p style={{ color:C.red, fontWeight:600, fontSize:13 }}>
            Error: {statusData?.error ?? 'Unknown error occurred.'}
          </p>
        </div>
      )}
    </div>
  );
};

// Processed Output Tab 
const OutputTab = ({ ready }) => {
  const videoRef                    = useRef();
  const [analytics, setAnalytics]   = useState([]);
  const [avgDwell,  setAvgDwell]    = useState(0);
  const [videoKey,  setVideoKey]    = useState(0);

  useEffect(() => {
    if (!ready) return;
    setVideoKey(k => k + 1);
    fetch(`${API}/api/video/analytics`)
      .then(r => r.json())
      .then(res => {
        setAnalytics(res.data || []);
        setAvgDwell(res.avg_dwell_sec || 0);
      })
      .catch(() => {});
  }, [ready]);

  const peakQueue   = analytics.length ? Math.max(...analytics.map(d => d.queue))  : 0;
  const totalUnique = analytics.length ? Math.max(...analytics.map(d => d.unique)) : 0;
  const durationSec = analytics.length ? analytics[analytics.length - 1].time      : 0;

  // Queue size distribution: how many seconds the video had each queue size
  const distData = useMemo(() => {
    const counts = {};
    analytics.forEach(d => { counts[d.queue] = (counts[d.queue] || 0) + 1; });
    return Object.entries(counts)
      .map(([k, v]) => ({ size: parseInt(k), seconds: v }))
      .sort((a, b) => a.size - b.size);
  }, [analytics]);

  if (!ready) {
    return (
      <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
        justifyContent:'center', minHeight:360, gap:16, animation:'fadeUp 0.3s ease',
        color:C.muted }}>
        <p style={{ fontSize:'3rem' }}>🎬</p>
        <p style={{ fontWeight:600, color:C.text }}>No processed video yet</p>
        <p style={{ fontSize:13 }}>Upload and process a video in the
          <strong style={{ color:C.primary }}> Upload Video</strong> tab first.</p>
      </div>
    );
  }

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1rem', animation:'fadeUp 0.3s ease' }}>

      {/* Header — title + download only */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <h2 style={{ fontSize:'1.1rem', fontWeight:700, color:C.text }}>Processed Output</h2>
        <a href={`${API}/api/video/output`}
          style={{ padding:'0.55rem 1.2rem', borderRadius:8, background:C.primaryDim,
            color:C.primary, fontWeight:600, fontSize:13, textDecoration:'none',
            border:`1px solid ${C.primary}` }}>
          ⬇ Download
        </a>
      </div>

      {/* Side-by-side: video left, analytics right */}
      <div style={{ display:'flex', gap:'1.2rem', alignItems:'flex-start' }}>

        {/* ── Left: video player ── */}
        <div style={{ flex:'0 0 58%', minWidth:0 }}>
          <div style={{ ...card(), padding:0, overflow:'hidden', background:'#000', borderRadius:12 }}>
            <video key={videoKey} ref={videoRef} controls
              style={{ width:'100%', display:'block' }}>
              <source src={`${API}/api/video/output`} type="video/mp4" />
              Your browser does not support the video element.
            </video>
          </div>
        </div>

        {/* ── Right: KPI cards + chart ── */}
        <div style={{ flex:1, minWidth:0, display:'flex', flexDirection:'column', gap:'0.85rem' }}>

          {/* 2×2 KPI grid */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'0.75rem' }}>
            {[
              { label:'Peak Queue',      value: peakQueue,         sub:'Max people at once',        accent: C.primary },
              { label:'Unique People',   value: totalUnique,       sub:'Distinct identities seen',  accent: C.green   },
              { label:'Avg Dwell Time',  value: `${avgDwell}s`,    sub:'Avg time spent in frame',   accent: C.amber   },
              { label:'Video Duration',  value: `${durationSec}s`, sub:'Total footage analysed',    accent: C.purple  },
            ].map(({ label, value, sub, accent }) => (
              <div key={label} style={{ ...card(), borderTop:`3px solid ${accent}`, padding:'1rem 1.1rem' }}>
                <p style={{ fontSize:10, color:C.muted, textTransform:'uppercase',
                  letterSpacing:1.2, marginBottom:4 }}>{label}</p>
                <p style={{ fontSize:'1.6rem', fontWeight:800, color:accent,
                  lineHeight:1, margin:'2px 0' }}>{value}</p>
                <p style={{ fontSize:11, color:C.dim }}>{sub}</p>
              </div>
            ))}
          </div>

          {/* Time-series chart */}
          {analytics.length > 0 && (
            <div style={{ ...card(), padding:'1rem 1.1rem' }}>
              <p style={{ fontSize:12, fontWeight:600, color:C.muted,
                textTransform:'uppercase', letterSpacing:1, marginBottom:'0.75rem' }}>
                Queue &amp; Unique People Over Time
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={analytics} margin={{ top:4, right:8, left:-16, bottom:0 }}>
                  <defs>
                    <linearGradient id="gVQ" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.primary} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={C.primary} stopOpacity={0}   />
                    </linearGradient>
                    <linearGradient id="gVU" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.green} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={C.green} stopOpacity={0}   />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.cardBorder} vertical={false} />
                  <XAxis dataKey="time" stroke={C.dim} fontSize={10}
                    tickFormatter={v => `${v}s`} tickMargin={4} />
                  <YAxis stroke={C.dim} fontSize={10} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ background:C.surface, border:`1px solid ${C.cardBorder}`,
                      borderRadius:10, color:C.text, fontSize:12 }}
                    labelFormatter={v => `Time: ${v}s`}
                  />
                  <Legend wrapperStyle={{ fontSize:11, color:C.muted, paddingTop:6 }} />
                  <Area type="monotone" dataKey="queue"  name="Queue Length"
                    stroke={C.primary} fill="url(#gVQ)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="unique" name="Unique People"
                    stroke={C.green}   fill="url(#gVU)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

        </div>
      </div>

      {/* Full-width: Queue Size Distribution bar chart */}
      {distData.length > 0 && (
        <div style={{ ...card(), padding:'1.2rem 1.4rem' }}>
          <p style={{ fontSize:12, fontWeight:700, color:C.muted,
            textTransform:'uppercase', letterSpacing:1.5, marginBottom:'0.25rem' }}>
            Queue Size Distribution
          </p>
          <p style={{ fontSize:12, color:C.dim, marginBottom:'1rem' }}>
            How many seconds of the video had each number of people in frame
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={distData} margin={{ top:4, right:20, left:0, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.cardBorder} vertical={false} />
              <XAxis
                dataKey="size"
                stroke={C.dim}
                fontSize={11}
                tickMargin={6}
                tickFormatter={v => v === 1 ? '1 person' : `${v} people`}
              />
              <YAxis stroke={C.dim} fontSize={11} allowDecimals={false}
                label={{ value:'seconds', angle:-90, position:'insideLeft',
                  fill:C.dim, fontSize:10, dy:30 }} />
              <Tooltip
                contentStyle={{ background:C.surface, border:`1px solid ${C.cardBorder}`,
                  borderRadius:10, color:C.text, fontSize:12 }}
                formatter={(v, _) => [`${v}s`, 'Duration']}
                labelFormatter={v => v === 1 ? '1 person in frame' : `${v} people in frame`}
              />
              <Bar dataKey="seconds" name="Duration (s)" radius={[6,6,0,0]}>
                {distData.map((entry, i) => {
                  const colours = [C.dim, C.primary, C.green, C.amber, C.purple, C.red];
                  return <Cell key={i} fill={colours[Math.min(i, colours.length - 1)]} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

    </div>
  );
};

// Root
export default function Dashboard() {
  const [tab,        setTab]        = useState('analytics');
  const [latest,     setLatest]     = useState({ queue_length:0, foot_traffic:0, unique_visitors:0 });
  const [history,    setHistory]    = useState([]);
  const [summary,    setSummary]    = useState({ peak_queue:0, avg_queue:0 });
  const [hourly,     setHourly]     = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [videoReady, setVideoReady] = useState(false);
  const [now,        setNow]        = useState(new Date());

  // Format history timestamps
  const formattedHistory = useMemo(() =>
    history.map(d => ({
      ...d,
      time: new Date(d.timestamp).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' }),
    })), [history]);

  const fetchAnalytics = useCallback(async () => {
    try {
      const [latestRes, historyRes, summaryRes, hourlyRes, embedsRes] = await Promise.all([
        fetch(`${API}/api/analytics/latest`),
        fetch(`${API}/api/analytics/history?limit=60`),
        fetch(`${API}/api/analytics/summary`),
        fetch(`${API}/api/analytics/hourly`),
        fetch(`${API}/api/embeddings`),
      ]);
      const [l, h, s, hr, e] = await Promise.all([
        latestRes.json(), historyRes.json(), summaryRes.json(),
        hourlyRes.json(), embedsRes.json(),
      ]);
      if (l && !l.message) setLatest(l);
      setHistory(h);
      setSummary(s);
      setHourly(hr);
      setEmbeddings(e);
    } catch { /* backend not ready yet */ }
  }, []);

  useEffect(() => {
    fetchAnalytics();
    const dataTimer  = setInterval(fetchAnalytics, 2000);
    const clockTimer = setInterval(() => setNow(new Date()), 1000);
    return () => { clearInterval(dataTimer); clearInterval(clockTimer); };
  }, [fetchAnalytics]);

  return (
    <>
      <GlobalStyle />
      <div style={{ minHeight:'100vh', background:C.bg, padding:'1.6rem 2rem' }}>

        {/* Header */}
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center',
          marginBottom:'1.8rem', flexWrap:'wrap', gap:12 }}>
          <div>
            <h1 style={{ fontSize:'1.5rem', fontWeight:800, color:C.text, letterSpacing:-0.5 }}>
              Privacy Analytics
            </h1>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:16 }}>
            <LiveBadge />
            <span style={{ fontSize:13, color:C.muted, fontVariantNumeric:'tabular-nums' }}>
              {now.toLocaleTimeString()}
            </span>
          </div>
        </div>

        {/* Tab nav */}
        <TabBar active={tab} onChange={setTab} />

        {/* Tab content */}
        {tab === 'analytics' && (
          <AnalyticsTab
            latest={latest}
            history={formattedHistory}
            summary={summary}
            hourly={hourly}
            embeddings={embeddings}
          />
        )}
        {tab === 'live'     && <LiveTab />}
        {tab === 'upload'   && (
          <UploadTab onProcessingDone={() => setVideoReady(true)} />
        )}
        {tab === 'output'   && <OutputTab ready={videoReady} />}

      </div>
    </>
  );
}
