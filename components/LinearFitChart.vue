<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  slope: { type: Number, default: 1.5 },
  intercept: { type: Number, default: 0.3 },
  noise: { type: Number, default: 1.2 },
  n: { type: Number, default: 30 },
  seed: { type: Number, default: 42 },
})

function seededRandom(seed) {
  let s = seed
  return () => { s = (s * 16807) % 2147483647; return s / 2147483647 }
}

const option = computed(() => {
  const rand = seededRandom(props.seed)
  const x = [], y = []
  for (let i = 0; i < props.n; i++) {
    const xi = rand() * 5
    const yi = props.slope * xi + props.intercept + (rand() - 0.5) * props.noise * 2
    x.push(+xi.toFixed(2))
    y.push(+yi.toFixed(2))
  }

  const xTy = x.reduce((s, xi, i) => s + xi * y[i], 0)
  const xTx = x.reduce((s, xi) => s + xi * xi, 0)
  const theta = xTy / xTx
  const yMean = y.reduce((s, v) => s + v, 0) / props.n
  const xMean = x.reduce((s, v) => s + v, 0) / props.n
  const b = yMean - theta * xMean

  const xMin = -0.2, xMax = 5.5

  // Residual lines: from (x_i, y_i) vertically to (x_i, theta*x_i + b)
  const residualLines = x.map((xi, i) => ({
    type: 'line',
    data: [[xi, y[i]], [xi, theta * xi + b]],
    showSymbol: false,
    lineStyle: { width: 1, color: 'rgba(251, 191, 36, 0.6)', type: 'dashed' },
    z: 1,
  }))

  return {
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    tooltip: { trigger: 'item' },
    xAxis: { type: 'value', name: 'x', min: xMin, max: xMax, splitLine: { lineStyle: { color: '#333' } }, axisLine: { lineStyle: { color: '#666' } }, axisLabel: { color: '#aaa' } },
    yAxis: { type: 'value', name: 'y', min: -1, max: 10, splitLine: { lineStyle: { color: '#333' } }, axisLine: { lineStyle: { color: '#666' } }, axisLabel: { color: '#aaa' } },
    series: [
      { type: 'scatter', data: x.map((xi, i) => [xi, y[i]]), symbolSize: 8, itemStyle: { color: '#60a5fa' } },
      {
        type: 'line', data: [[xMin, theta * xMin + b], [xMax, theta * xMax + b]],
        showSymbol: false, lineStyle: { width: 2.5, color: '#f87171' },
        label: { show: true, formatter: `y = ${theta.toFixed(2)}x + ${b.toFixed(2)}`, position: 'end', color: '#f87171', fontSize: 12 },
      },
      ...residualLines,
    ]
  }
})
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
</template>
