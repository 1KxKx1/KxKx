{{/* Common template helpers. */}}

{{- define "kubeai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "kubeai.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "kubeai.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "kubeai.labels" -}}
app.kubernetes.io/name: {{ include "kubeai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/component: inference-gateway
app.kubernetes.io/part-of: kubeai-platform
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "kubeai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kubeai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "kubeai.serviceAccountName" -}}
{{- if .Values.argo.enabled -}}
{{- default (printf "%s-workflow" (include "kubeai.fullname" .)) .Values.argo.serviceAccount.name -}}
{{- else -}}
default
{{- end -}}
{{- end -}}

{{- define "kubeai.modelsConfig" -}}
memoryBudgetMib: {{ .Values.memoryBudgetMib }}
models:
{{ toYaml .Values.models | indent 2 }}
{{- end -}}
