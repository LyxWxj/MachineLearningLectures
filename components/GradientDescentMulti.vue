<script setup>
import { ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, CanvasRenderer])

// f(x) = x^4 - 4x^3 + 2x^2 + 4x
// Two local minima, one local maximum
function f(x) { return x ** 4 - 4 * x ** 3 + 2 * x ** 2 + 4 * x }
function fPrime(x) { return 4 * x ** 3 - 12 * x ** 2 + 4 * x + 4 }

const x0 = ref(3.5)
const lr = 0.15
const maxSteps = 30

const pathData = computed(() => {
  const points = [[x0.value, f(x0.value)]]
  let x = x0.value
  for (let i = 0; i < maxSteps; i++) {
    x = x - lr * fPrime(x)
    x = Math.max(-2, Math.min(5, x)) // clamp to visible range
    points.push([+x.toFixed(4), +f(x).toFixed(4)])
    if (Math.abs(fPrime(x)) < 0.01) break
  }
  return points
})

const convergedTo = computed(() => {
  const last = pathData.value[pathData.value.length - 1][0]
  if (last < 0.8) return 'Local min (left)'
  if (last > 1.8) return 'Local min (right)'
  return 'Local max'
})

const convergedColor = computed(() => {
  const last = pathData.value[pathData.value.length - 1][0]
  if (last < 0.8) return '#34d399'
  if (last > 1.8) return '#f97316'
  return '#ef4444'
})

const xMin = -1.5
const xMax = 4.5
const curveStep = 0.02

const curveData = computed(() => {
  const data = []
  for (let x = xMin; x <= xMax; x += curveStep) {
    data.push([+x.toFixed(3), f(x)])
  }
  return data
})

// Critical points for marking
const critPoints = [
  { x: -0.414, label: 'local min' },
  { x: 1, label: 'local max' },
  { x: 2.414, label: 'local min' },
]

const option = computed(() => ({
  animation: false,
  grid: { left: 50, right: 30, top: 20, bottom: 40 },
  xAxis: {
    type: 'value', min: xMin, max: xMax,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  yAxis: {
    type: 'value', min: -5, max: 8,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  series: [
    {
      name: 'f(x)',
      type: 'line',
      data: curveData.value,
      showSymbol: false,
      lineStyle: { width: 2, color: '#60a5fa' },
      smooth: true,
    },
    {
      name: 'descent path',
      type: 'line',
      data: pathData.value,
      showSymbol: true,
      symbolSize: 6,
      lineStyle: { width: 1.5, color: '#f87171', type: 'dashed' },
      itemStyle: { color: '#f87171' },
      label: {
        show: true,
        formatter: (p) => {
          if (p.dataIndex === 0) return 'x₀'
          if (p.dataIndex === pathData.value.length - 1) return `x${p.dataIndex}`
          return ''
        },
        position: 'top',
        color: '#f87171',
        fontSize: 10,
      },
    },
    {
      name: 'critical points',
      type: 'scatter',
      data: critPoints.map(p => [p.x, f(p.x)]),
      symbolSize: 10,
      itemStyle: { color: '#fbbf24', borderColor: '#fff', borderWidth: 1 },
      label: {
        show: true,
        formatter: (p) => critPoints[p.dataIndex].label,
        position: 'bottom',
        color: '#fbbf24',
        fontSize: 9,
      },
    },
  ],
}))
</script>

<template>
  <div class="flex flex-col w-full h-full">
    <!-- Control bar -->
    <div class="control-bar flex items-center gap-4 py-2 px-4 rounded-lg">
      <span class="text-xs text-gray-400">x₀</span>
      <input
        type="range"
        :min="xMin"
        :max="xMax"
        step="0.1"
        v-model.number="x0"
        class="slider-x0"
      />
      <span class="font-mono text-sm text-blue-400 font-bold w-10 text-center">{{ x0.toFixed(1) }}</span>
      <span class="text-xs font-semibold" :style="{ color: convergedColor }">→ {{ convergedTo }}</span>
    </div>

    <VChart :option="option" autoresize class="flex-1" />
  </div>
</template>

<style scoped>
.control-bar {
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.08), rgba(249, 115, 22, 0.08));
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.slider-x0 {
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  height: 5px;
  border-radius: 3px;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  outline: none;
  cursor: pointer;
}
.slider-x0::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #8b5cf6;
  box-shadow: 0 0 6px rgba(139, 92, 246, 0.7);
  transition: box-shadow 0.2s;
}
.slider-x0::-webkit-slider-thumb:hover {
  box-shadow: 0 0 12px rgba(139, 92, 246, 1);
}
.slider-x0::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #8b5cf6;
  border: none;
  box-shadow: 0 0 6px rgba(139, 92, 246, 0.7);
}
</style>
