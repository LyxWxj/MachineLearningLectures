<script setup>
import { ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, CanvasRenderer])

// f(x) = (x-2)^2 + 1
function f(x) { return (x - 2) ** 2 + 1 }
function fPrime(x) { return 2 * (x - 2) }

const lr = ref(0.3)
const startX = -0.5
const maxSteps = 15

const convergenceThreshold = 0.005

const pathData = computed(() => {
  const points = [[startX, f(startX)]]
  let x = startX
  for (let i = 0; i < maxSteps; i++) {
    x = x - lr.value * fPrime(x)
    if (Math.abs(x) > 100) break // diverged
    points.push([+x.toFixed(4), +f(x).toFixed(4)])
    if (Math.abs(fPrime(x)) < convergenceThreshold) break
  }
  return points
})

const status = computed(() => {
  const pts = pathData.value
  if (pts.length < 2) return { text: '', color: '#aaa' }
  const last = pts[pts.length - 1][0]
  if (Math.abs(last) > 50) return { text: 'Diverged!', color: '#ef4444' }
  if (Math.abs(last - 2) < 0.1 && pts.length <= 6) return { text: 'Converged fast', color: '#34d399' }
  if (Math.abs(last - 2) < 0.1) return { text: 'Converged (slow)', color: '#fbbf24' }
  if (pts.length >= maxSteps) return { text: 'Not converged yet', color: '#f97316' }
  return { text: '', color: '#aaa' }
})

const xMin = -2
const xMax = 5
const curveStep = 0.05

const curveData = computed(() => {
  const data = []
  for (let x = xMin; x <= xMax; x += curveStep) {
    data.push([+x.toFixed(3), f(x)])
  }
  return data
})

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
    type: 'value', min: 0, max: 12,
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
      name: 'path',
      type: 'line',
      data: pathData.value,
      showSymbol: true,
      symbolSize: 6,
      lineStyle: { width: 1.5, color: '#f87171', type: 'dashed' },
      itemStyle: { color: '#f87171' },
      label: {
        show: true,
        formatter: (p) => p.dataIndex === 0 ? 'x₀' : `x${p.dataIndex}`,
        position: 'top',
        color: '#f87171',
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
      <span class="text-xs text-gray-400">η</span>
      <input
        type="range"
        min="0.01"
        max="1.2"
        step="0.01"
        v-model.number="lr"
        class="slider-lr"
      />
      <span class="font-mono text-sm text-orange-400 font-bold w-14 text-center">{{ lr.toFixed(2) }}</span>
      <span class="text-xs font-semibold" :style="{ color: status.color }">{{ status.text }}</span>
      <span class="text-xs text-gray-500 ml-auto">{{ pathData.length - 1 }} steps</span>
    </div>

    <VChart :option="option" autoresize class="flex-1" />
  </div>
</template>

<style scoped>
.control-bar {
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.08), rgba(249, 115, 22, 0.08));
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.slider-lr {
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  height: 5px;
  border-radius: 3px;
  background: linear-gradient(90deg, #1e40af, #f97316, #ef4444);
  outline: none;
  cursor: pointer;
}
.slider-lr::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #f97316;
  box-shadow: 0 0 6px rgba(249, 115, 22, 0.7);
  transition: box-shadow 0.2s;
}
.slider-lr::-webkit-slider-thumb:hover {
  box-shadow: 0 0 12px rgba(249, 115, 22, 1);
}
.slider-lr::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #f97316;
  border: none;
  box-shadow: 0 0 6px rgba(249, 115, 22, 0.7);
}
</style>
