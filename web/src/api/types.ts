export interface TaskSummary {
  task_id: string;
  status: string;
  harness?: string | null;
  model?: string | null;
  num_samples: number;
  completed_sessions: number;
  errored_sessions?: number;
  mean_reward?: number | null;
  mean_traces?: number | null;
  mean_completions?: number | null;
  created_at?: number | null;
  updated_at?: number | null;
  save_dir_path?: string;
  session_files?: string[];
  source?: string;
}

export interface SessionTiming {
  register_to_init_queue_ms?: number;
  init_ms?: number;
  run_ms?: number;
  postrun_ms?: number;
  [key: string]: number | Record<string, number> | undefined;
}

export interface SessionSummary {
  session_id: string;
  task_id?: string;
  status: string;
  node_id?: string | null;
  reward?: number | null;
  timing?: SessionTiming;
  error?: string | null;
  file_path?: string | null;
}

export interface Trace {
  prompt_ids?: number[];
  response_ids?: number[];
  prompt_messages?: { role: string; content: any }[];
  response_messages?: { role: string; content: any }[];
  reward?: number | null;
  finish_reason?: string | null;
  response_logprobs?: any[] | null;
  metadata?: Record<string, any>;
}

export interface TrajectoryPayload {
  session_id: string;
  status?: string | null;
  metadata: Record<string, any>;
  traces: Trace[];
  error?: string | null;
}

export interface CompletionRecord {
  completion_id: string;
  timestamp?: string;
  session_id?: string;
  task_id?: string | null;
  api_type?: string | null;
  model_requested?: string | null;
  model_used?: string | null;
  original_request?: Record<string, any>;
  transformed_request?: Record<string, any>;
  request?: Record<string, any>;
  response?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface EvaluationResult {
  session_id: string;
  outcome_reward?: number | null;
  strategy?: string | null;
  report?: any;
  patch_path?: string | null;
  trace_rewards?: (number | null)[];
  raw?: any;
}

export interface TopologyPayload {
  rollout: {
    url: string;
    host: string;
    port: number;
    save_dir: string;
    reachable: boolean;
    health: any;
    status: any;
  };
  gateways: {
    node_id: string;
    host: string;
    port: number;
    gateway_url: string;
    model_served: string;
    engine: string;
    inference_base_url: string;
    max_init_workers: number;
    max_run_workers: number;
    max_postrun_workers: number;
    reachable: boolean;
    health: any;
  }[];
}
