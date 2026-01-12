'use client';

import { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Run {
  run_uid: string;
  true_fault: number;
  simulationRun: number;
}

const VARIABLE_DESCRIPTIONS: Record<string, string> = {
  'xmeas_1': 'A feed stream',
  'xmeas_2': 'D feed stream',
  'xmeas_3': 'E feed stream',
  'xmeas_4': 'Total fresh feed stripper',
  'xmeas_5': 'Recycle flow into rxtr',
  'xmeas_6': 'Reactor feed rate',
  'xmeas_7': 'Reactor pressure',
  'xmeas_8': 'Reactor level',
  'xmeas_9': 'Reactor temp',
  'xmeas_10': 'Purge rate',
  'xmeas_11': 'Separator temp',
  'xmeas_12': 'Separator level',
  'xmeas_13': 'Separator pressure',
  'xmeas_14': 'Separator underflow',
  'xmeas_15': 'Stripper level',
  'xmeas_16': 'Stripper pressure',
  'xmeas_17': 'Stripper underflow',
  'xmeas_18': 'Stripper temperature',
  'xmeas_19': 'Stripper steam flow',
  'xmeas_20': 'Compressor work',
  'xmeas_21': 'Reactor cooling water outlet temp',
  'xmeas_22': 'Condenser cooling water outlet temp',
  'xmeas_23': 'Composition of A rxtr feed',
  'xmeas_24': 'Composition of B rxtr feed',
  'xmeas_25': 'Composition of C rxtr feed',
  'xmeas_26': 'Composition of D rxtr feed',
  'xmeas_27': 'Composition of E rxtr feed',
  'xmeas_28': 'Composition of F rxtr feed',
  'xmeas_29': 'Composition of A purge',
  'xmeas_30': 'Composition of B purge',
  'xmeas_31': 'Composition of C purge',
  'xmeas_32': 'Composition of D purge',
  'xmeas_33': 'Composition of E purge',
  'xmeas_34': 'Composition of F purge',
  'xmeas_35': 'Composition of G purge',
  'xmeas_36': 'Composition of H purge',
  'xmeas_37': 'Composition of D product',
  'xmeas_38': 'Composition of E product',
  'xmeas_39': 'Composition of F product',
  'xmeas_40': 'Composition of G product',
  'xmeas_41': 'Composition of H product',
  'xmv_1': 'D feed flow valve',
  'xmv_2': 'E feed flow valve',
  'xmv_3': 'A feed flow valve',
  'xmv_4': 'Total feed flow stripper valve',
  'xmv_5': 'Compressor recycle valve',
  'xmv_6': 'Purge valve',
  'xmv_7': 'Separator pot liquid flow valve',
  'xmv_8': 'Stripper liquid product flow valve',
  'xmv_9': 'Stripper steam valve',
  'xmv_10': 'Reactor cooling water flow valve',
  'xmv_11': 'Condenser cooling water flow valve',
  'xmv_12': 'Agitator speed',
};

type WebSocketEvent = 
  | { type: 'start'; run_uid: string; simulationRun: number; true_fault: number; is_faulty_run: number; eval_every_n_samples: number }
  | { type: 'tick'; run_uid: string; sample: number; state: 'NORMAL' | 'SUSPECT' | 'ALERT'; alert_type: string; evaluated: boolean; xgb?: { pred_fault: number; conf: number | null; raw: number; stable: number }; ae?: { recon_error: number | null; raw: number; stable: number } }
  | { type: 'alert'; run_uid: string; sample: number; state: string; alert_type: string; pred_fault?: number | null }
  | { type: 'rca'; run_uid: string; sample: number; model: string; top_drivers: Array<{ feature: string; contribution: number }> }
  | { type: 'done'; run_uid: string }
  | { type: 'error'; message: string };

export default function Home() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [sessionId, setSessionId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [events, setEvents] = useState<WebSocketEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Load runs when page loads
  useEffect(() => {
    fetch(`${API_URL}/runs`)
      .then((res) => res.json())
      .then((data) => setRuns(data))
      .catch((err) => {
        console.error('Failed to fetch runs:', err);
        setError('Failed to load runs. Make sure the backend is running.');
      });
  }, []);

  // Handle WebSocket connection
  useEffect(() => {
    if (!sessionId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setIsConnected(false);
      }
      return;
    }

    setEvents([]);
    setError('');

    let wsUrl = API_URL.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
    wsUrl = wsUrl.replace(/\/$/, '');
    const ws = new WebSocket(`${wsUrl}/ws/${sessionId}`);
    
    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketEvent;
        setEvents((prev) => [...prev, data]);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  const handleStartMonitoring = async () => {
    if (!selectedRun) return;
    
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_URL}/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_uid: selectedRun, tick_interval_s: 0.0 }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create session');
      }

      const data = await response.json();
      setSessionId(data.session_id);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to create session');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStop = () => {
    setSessionId('');
  };

  const handleRunChange = (runUid: string) => {
    setSelectedRun(runUid);
    if (sessionId) {
      handleStop();
    }
  };

  const handleClearEvents = () => {
    setEvents([]);
  };

  // Find different event types
  const startEvent = events.find((e) => e.type === 'start');
  const tickEvents = events.filter((e) => e.type === 'tick');
  const latestTick = tickEvents.length > 0 ? tickEvents[tickEvents.length - 1] : null;
  const alerts = events.filter((e) => e.type === 'alert');
  const rcaEvents = events.filter((e) => e.type === 'rca');
  const doneEvent = events.find((e) => e.type === 'done');

  // Prepare chart data
  const chartData: Array<{ sample: number; xgbConf: number | null; aeReconError: number | null; state: string }> = [];
  for (const event of events) {
    if (event.type === 'tick' && event.evaluated) {
      chartData.push({
        sample: event.sample,
        xgbConf: event.xgb?.conf ?? null,
        aeReconError: event.ae?.recon_error ?? null,
        state: event.state,
      });
    }
  }

  // Get state color
  const getStateColor = (state: string) => {
    if (state === 'NORMAL') return 'bg-green-500';
    if (state === 'SUSPECT') return 'bg-yellow-500';
    if (state === 'ALERT') return 'bg-red-500';
    return 'bg-gray-500';
  };

  const getStateTextColor = (state: string) => {
    if (state === 'NORMAL') return 'text-green-700';
    if (state === 'SUSPECT') return 'text-yellow-700';
    if (state === 'ALERT') return 'text-red-700';
    return 'text-gray-700';
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Live Fault Monitor
          </h1>
          <p className="text-gray-600 text-lg mb-6">
            Simulates a real-time hybrid machine learning model to detect, classify, and assist root cause sourcing of manufacturing faults on Tennessee Eastman Process Simulated Dataset.
          </p>
        </div>

        {/* Decision Path Diagram */}
        <div className="mb-8 bg-white rounded-lg shadow-md border border-gray-200 p-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">
            Decision Path
          </h2>
          
          <div className="max-w-4xl mx-auto">
            {/* Start Box */}
            <div className="flex justify-center mb-4">
              <div className="bg-blue-100 border-2 border-blue-300 rounded-lg px-6 py-3">
                <div className="text-center font-semibold text-blue-900">
                  Process Sample Data
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center mb-4">
              <svg className="w-6 h-8" fill="none" viewBox="0 0 24 32">
                <path d="M12 0 L12 24 M12 24 L6 18 M12 24 L18 18" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-600"/>
              </svg>
            </div>

            {/* XGBoost Model Box */}
            <div className="flex justify-center mb-4">
              <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4 max-w-md">
                <div className="font-semibold text-blue-900 mb-3 text-center">
                  XGBoost Model
                </div>
                <div className="text-sm text-gray-700">
                  <div>• Confidence ≥ 0.8</div>
                  <div>• Requires 3 consecutive sample detections</div>
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center mb-4">
              <svg className="w-6 h-8" fill="none" viewBox="0 0 24 32">
                <path d="M12 0 L12 24 M12 24 L6 18 M12 24 L18 18" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600"/>
              </svg>
            </div>

            {/* Decision Diamond */}
            <div className="flex justify-center mb-4">
              <div className="w-32 h-32 transform rotate-45 bg-blue-200 border-2 border-blue-400 rounded-lg flex items-center justify-center">
                <div className="transform -rotate-45 text-sm font-semibold text-blue-900 text-center px-3">
                  XGBoost Detects?
                </div>
              </div>
            </div>

            {/* Branch Arrows */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {/* Yes Branch */}
              <div className="flex flex-col items-center">
                <div className="text-sm text-red-700 font-semibold mb-3">Yes</div>
                <svg className="w-8 h-10 mb-3" fill="none" viewBox="0 0 32 40">
                  <path d="M16 0 L16 30 M16 30 L8 22 M16 30 L24 22" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-red-600"/>
                </svg>
                <div className="bg-red-100 border-2 border-red-400 rounded-lg p-4 text-center w-full">
                  <div className="font-semibold text-red-900 text-lg">
                    Known Fault
                  </div>
                </div>
              </div>

              {/* No Branch */}
              <div className="flex flex-col items-center">
                <div className="text-sm text-gray-700 font-semibold mb-3">No</div>
                <svg className="w-8 h-10 mb-3" fill="none" viewBox="0 0 32 40">
                  <path d="M16 0 L16 30 M16 30 L8 22 M16 30 L24 22" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-600"/>
                </svg>
                <div className="bg-gray-100 border-2 border-gray-400 rounded-lg p-3 text-center w-full">
                  <div className="text-sm font-medium text-gray-700">
                    Continue to Autoencoder
                  </div>
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center mb-4">
              <svg className="w-6 h-8" fill="none" viewBox="0 0 24 32">
                <path d="M12 0 L12 24 M12 24 L6 18 M12 24 L18 18" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-600"/>
              </svg>
            </div>

            {/* Autoencoder Model Box */}
            <div className="flex justify-center mb-4">
              <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4 max-w-md">
                <div className="font-semibold text-purple-900 mb-3 text-center">
                  Autoencoder Model
                </div>
                <div className="text-sm text-gray-700">
                  <div>• Reconstruction Error ≥ 0.78</div>
                  <div>• Requires 2 consecutive detections in 10-sample windows</div>
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center mb-4">
              <svg className="w-6 h-8" fill="none" viewBox="0 0 24 32">
                <path d="M12 0 L12 24 M12 24 L6 18 M12 24 L18 18" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600"/>
              </svg>
            </div>

            {/* Decision Diamond 2 */}
            <div className="flex justify-center mb-4">
              <div className="w-32 h-32 transform rotate-45 bg-purple-200 border-2 border-purple-400 rounded-lg flex items-center justify-center">
                <div className="transform -rotate-45 text-sm font-semibold text-purple-900 text-center px-3">
                  Autoencoder Detects?
                </div>
              </div>
            </div>

            {/* Final Branch Arrows */}
            <div className="grid grid-cols-2 gap-6">
              {/* Yes Branch */}
              <div className="flex flex-col items-center">
                <div className="text-sm text-red-700 font-semibold mb-3">Yes</div>
                <svg className="w-8 h-10 mb-3" fill="none" viewBox="0 0 32 40">
                  <path d="M16 0 L16 30 M16 30 L8 22 M16 30 L24 22" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-red-600"/>
                </svg>
                <div className="bg-red-100 border-2 border-red-400 rounded-lg p-4 text-center w-full">
                  <div className="font-semibold text-red-900 text-lg">
                    Anomaly Unclassified
                    <div className="text-xs font-normal mt-1">(Unknown Fault)</div>
                  </div>
                </div>
              </div>

              {/* No Branch */}
              <div className="flex flex-col items-center">
                <div className="text-sm text-green-700 font-semibold mb-3">No</div>
                <svg className="w-8 h-10 mb-3" fill="none" viewBox="0 0 32 40">
                  <path d="M16 0 L16 30 M16 30 L8 22 M16 30 L24 22" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-green-600"/>
                </svg>
                <div className="bg-green-100 border-2 border-green-400 rounded-lg p-4 text-center w-full">
                  <div className="font-semibold text-green-900 text-lg">
                    Fault Free
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-800 rounded">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Control Panel
              </h2>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Run
                </label>
                <select
                  value={selectedRun}
                  onChange={(e) => handleRunChange(e.target.value)}
                  disabled={!!sessionId}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                >
                  <option value="">-- Select a run --</option>
                  {runs.map((run) => (
                    <option key={run.run_uid} value={run.run_uid}>
                      {run.run_uid} (Fault: {run.true_fault}, Sim: {run.simulationRun})
                    </option>
                  ))}
                </select>
                <div className="mt-2 text-xs text-gray-600">
                  <div><strong>FF</strong>: Fault-free data</div>
                  <div><strong>F1-F14</strong>: Known fault modes (detected by XGBoost Model)</div>
                  <div><strong>F16-F20</strong>: Unknown root cause faults</div>
                  <div className="mt-2 pt-2 border-t border-gray-300 italic">
                    * All faults are inserted at sample 20
                  </div>
                </div>
              </div>

              <div className="mb-6">
                {!sessionId ? (
                  <button
                    onClick={handleStartMonitoring}
                    disabled={!selectedRun || loading}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold rounded-lg"
                  >
                    {loading ? 'Starting...' : 'Start Monitoring'}
                  </button>
                ) : (
                  <button
                    onClick={handleStop}
                    className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg"
                  >
                    Stop Monitoring
                  </button>
                )}
              </div>

              <div className="pt-6 border-t border-gray-200">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Connection Status
                </h3>
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${sessionId ? 'bg-green-500' : 'bg-gray-400'}`} />
                  <span className="text-sm text-gray-600">
                    {sessionId ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Dashboard */}
          <div className="lg:col-span-2">
            {sessionId ? (
              <div className="space-y-6">
                {/* Header */}
                <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h2 className="text-2xl font-semibold text-gray-900">
                        Monitoring Dashboard
                      </h2>
                      {startEvent && startEvent.type === 'start' && (
                        <p className="text-sm text-gray-600 mt-1">
                          Run: {startEvent.run_uid} | Fault: {startEvent.true_fault} | Sim: {startEvent.simulationRun}
                        </p>
                      )}
                    </div>
                    <div className="flex items-center space-x-4">
                      {latestTick && latestTick.type === 'tick' && (
                        <div className="flex items-center space-x-2">
                          <div className={`w-4 h-4 rounded-full ${getStateColor(latestTick.state)}`} />
                          <span className={`font-semibold ${getStateTextColor(latestTick.state)}`}>
                            {latestTick.state}
                          </span>
                        </div>
                      )}
                      {doneEvent && (
                        <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                          Complete
                        </span>
                      )}
                    </div>
                  </div>

                  {latestTick && latestTick.type === 'tick' && (
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                        <div className="text-xs text-gray-600">Sample</div>
                        <div className="text-2xl font-bold text-gray-900">
                          {latestTick.sample}
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                        <div className="text-xs text-gray-600">Alert Type</div>
                        <div className="text-sm font-semibold text-gray-900">
                          {latestTick.alert_type || 'NONE'}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Charts */}
                <div className="space-y-6">
                  {/* XGB Confidence Chart */}
                  <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-xl font-semibold text-gray-900">XGBoost Confidence Over Time</h3>
                      <button
                        onClick={handleClearEvents}
                        className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg"
                      >
                        Clear
                      </button>
                    </div>
                    {chartData.length === 0 ? (
                      <div className="h-64 flex items-center justify-center text-gray-500">
                        No data available yet. Waiting for evaluated ticks...
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="sample" />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <ReferenceLine y={0.8} stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" label="Threshold" />
                          <Line type="monotone" dataKey="xgbConf" name="XGB Confidence" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>

                  {/* AE Reconstruction Error Chart */}
                  <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-4">Autoencoder Reconstruction Error Over Time</h3>
                    {chartData.length === 0 ? (
                      <div className="h-64 flex items-center justify-center text-gray-500">
                        No data available yet. Waiting for evaluated ticks...
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="sample" />
                          <YAxis />
                          <Tooltip />
                          <ReferenceLine y={0.78} stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" label="Threshold" />
                          <Line type="monotone" dataKey="aeReconError" name="AE Recon Error" stroke="#10b981" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>

                {/* Alerts */}
                {alerts.length > 0 && (
                  <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-4">
                      Alerts ({alerts.length})
                    </h3>
                    <div className="space-y-3">
                      {alerts.map((alert, idx) => {
                        if (alert.type !== 'alert') return null;
                        return (
                          <div
                            key={idx}
                            className={`p-4 rounded-lg border-l-4 ${
                              alert.alert_type === 'KNOWN_FAULT'
                                ? 'bg-red-50 border-red-400'
                                : 'bg-yellow-50 border-yellow-400'
                            }`}
                          >
                            <div className="font-semibold text-gray-900">
                              {alert.alert_type.replace('_', ' ')}
                            </div>
                            <div className="text-sm text-gray-600 mt-1">
                              Sample: {alert.sample} | State: {alert.state}
                            </div>
                            {alert.pred_fault !== null && alert.pred_fault !== undefined && (
                              <div className="text-sm text-gray-600">
                                Predicted Fault: {alert.pred_fault}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* RCA Results */}
                {rcaEvents.length > 0 && (
                  <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-4">
                      Root Cause Analysis ({rcaEvents.length})
                    </h3>
                    <div className="space-y-6">
                      {rcaEvents.map((rca, idx) => {
                        if (rca.type !== 'rca') return null;
                        
                        const top5Drivers = rca.top_drivers.slice(0, 5);
                        const contributions = top5Drivers.map(d => Math.abs(d.contribution));
                        const maxContribution = Math.max(...contributions, 1);

                        return (
                          <div key={idx} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                            <div className="mb-3">
                              <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                                {rca.model.toUpperCase()}
                              </span>
                              <span className="ml-3 text-sm text-gray-600">
                                Sample: {rca.sample}
                              </span>
                            </div>
                            <div>
                              <h4 className="text-sm font-semibold text-gray-700 mb-3">
                                Top Contributing Features:
                              </h4>
                              <div className="space-y-3">
                                {top5Drivers.map((driver, driverIdx) => {
                                  const contribution = driver.contribution;
                                  const absContribution = Math.abs(contribution);
                                  const percentage = (absContribution / maxContribution) * 100;
                                  const isPositive = contribution >= 0;
                                  const description = VARIABLE_DESCRIPTIONS[driver.feature.toLowerCase()] || '';

                                  return (
                                    <div key={driverIdx} className="space-y-1">
                                      <div className="flex items-center justify-between mb-1">
                                        <span className="text-sm text-gray-900">
                                          <span className="font-mono">{driver.feature}</span>
                                          {description && <span className="ml-2 text-gray-600">- {description}</span>}
                                        </span>
                                        <span className={`text-sm font-semibold ${isPositive ? 'text-blue-600' : 'text-red-600'}`}>
                                          {contribution.toFixed(4)}
                                        </span>
                                      </div>
                                      <div className="w-full bg-gray-200 rounded-full h-6">
                                        <div
                                          className={`h-full rounded-full ${isPositive ? 'bg-blue-500' : 'bg-red-500'}`}
                                          style={{ width: `${percentage}%` }}
                                        />
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-md border border-gray-200 p-12 text-center">
                <svg
                  className="mx-auto h-24 w-24 text-gray-400 mb-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  No Active Session
                </h3>
                <p className="text-gray-600">
                  Select a run and click &quot;Start Monitoring&quot; to begin
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
