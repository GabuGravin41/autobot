/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export enum BrowserMode {
  HUMAN_PROFILE = 'human_profile',
}

export enum AdapterPolicy {
  STRICT = 'strict',
  BALANCED = 'balanced',
  TRUSTED = 'trusted',
}

export interface TaskStep {
  action: string;
  args: Record<string, any>;
  description: string;
  save_as?: string;
  condition?: string;
  retries?: number;
  continue_on_error?: boolean;
  status?: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  screenshot?: string;
  target_node?: string;
}

export interface WorkflowPlan {
  id: string;
  name: string;
  description: string;
  steps: TaskStep[];
  topic?: string;
}

export interface RunHistory {
  id: string;
  planName: string;
  timestamp: string;
  status: 'success' | 'failed' | 'running';
  stepsCompleted: number;
  totalSteps: number;
  artifacts: Record<string, any>;
  screenshots: string[];
  logs: string[];
}

export interface Adapter {
  name: string;
  description: string;
  actions: string[];
  telemetry?: {
    calls: number;
    errors: number;
    lastUsed: string;
  };
}

export interface LLMModel {
  id: string;
  name: string;
  provider: string;
  isCustom?: boolean;
}

export interface UserProfile {
  name: string;
  email: string;
  avatar: string;
  role: string;
}

export interface AppState {
  browserMode: BrowserMode;
  adapterPolicy: AdapterPolicy;
  activeRun?: RunHistory;
  runs: RunHistory[];
  isAutonomous: boolean;
  theme: 'violet' | 'emerald' | 'blue' | 'amber';
  models: LLMModel[];
  selectedModelId: string;
  user: UserProfile;
}
