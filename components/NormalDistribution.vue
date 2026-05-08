<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, BarChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer])

const props = defineProps({
  mu: { type: Number, default: 0 },
  sigma: { type: Number, default: 1 },
  showSamples: { type: Boolean, default: false },
})

function normalPDF(x, mu, sigma) {
  const z = (x - mu) / sigma
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI))
}

const step = 0.05
const xMin = -4
const xMax = 4

const curveData = computed(() => {
  const data = []
  for (let x = xMin; x <= xMax; x += step) {
    data.push([+x.toFixed(3), normalPDF(x, props.mu, props.sigma)])
  }
  return data
})

// Generate histogram data for samples
const histogramData = computed(() => {
  if (!props.showSamples) return []
  const samples = []
  for (let i = 0; i < 500; i++) {
    // Box-Muller transform
    const u1 = Math.random()
    const u2 = Math.random()
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    samples.push(props.mu + props.sigma * z)
  }
  const bins = 30
  const binWidth = (xMax - xMin) / bins
  const counts = new Array(bins).fill(0)
  samples.forEach(s => {
    const idx = Math.floor((s - xMin) / binWidth)
    if (idx >= 0 && idx < bins) counts[idx]++
  })
  return counts.map((c, i) => [xMin + (i + 0.5) * binWidth, c / (500 * binWidth)])
})

const option = computed(() => ({
  animation: false,
  grid: { left: 50, right: 30, top: 10, bottom: 40 },
  xAxis: {
    type: 'value',
    min: xMin,
    max: xMax,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  yAxis: {
    type: 'value',
    min: 0,
    max: 0.5,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  series: [
    {
      type: 'line',
      data: curveData.value,
      showSymbol: false,
      lineStyle: { width: 2.5, color: '#60a5fa' },
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(96, 165, 250, 0.35)' },
            { offset: 1, color: 'rgba(96, 165, 250, 0.02)' },
          ],
        },
      },
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: '#f87171', type: 'solid', width: 1.5 },
        data: [{ xAxis: props.mu }],
        label: {
          show: true,
          formatter: 'μ',
          color: '#f87171',
          fontSize: 14,
          position: 'start',
        },
      },
    },
    ...(props.showSamples ? [{
      type: 'bar',
      data: histogramData.value,
      barWidth: '80%',
      itemStyle: { color: 'rgba(251, 191, 36, 0.4)', borderColor: 'rgba(251, 191, 36, 0.6)', borderWidth: 1 },
    }] : []),
  ],
}))
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
</template>
