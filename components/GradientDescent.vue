<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, CanvasRenderer])

// f(x) = (x-2)^2 + 1, minimum at x=2
function f(x) { return (x - 2) ** 2 + 1 }
function fPrime(x) { return 2 * (x - 2) }

const learningRate = 0.3
const startX = -0.5
const steps = 8

const pathData = computed(() => {
  const points = [[startX, f(startX)]]
  let x = startX
  for (let i = 0; i < steps; i++) {
    x = x - learningRate * fPrime(x)
    points.push([+x.toFixed(4), +f(x).toFixed(4)])
  }
  return points
})

const xMin = -1.5
const xMax = 4.5
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
    max: 12,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  series: [
    {
      type: 'line',
      data: curveData.value,
      showSymbol: false,
      lineStyle: { width: 2, color: '#60a5fa' },
      smooth: true,
    },
    {
      type: 'line',
      data: pathData.value,
      showSymbol: true,
      symbolSize: 8,
      lineStyle: { width: 1.5, color: '#f87171', type: 'dashed' },
      itemStyle: { color: '#f87171' },
      label: {
        show: true,
        formatter: (p) => `x${p.dataIndex}=${p.data[0].toFixed(2)}`,
        position: 'top',
        color: '#f87171',
        fontSize: 10,
      },
    },
  ],
}))
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
</template>
