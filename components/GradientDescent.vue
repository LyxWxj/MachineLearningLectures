<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

// f(x) = (x-2)^2 + 1, minimum at x=2
function f(x) { return (x - 2) ** 2 + 1 }
function fPrime(x) { return 2 * (x - 2) }

const learningRate = 0.3
const steps = 8

function computePath(startX) {
  const points = [[startX, f(startX)]]
  let x = startX
  for (let i = 0; i < steps; i++) {
    x = x - learningRate * fPrime(x)
    points.push([+x.toFixed(4), +f(x).toFixed(4)])
  }
  return points
}

const pathLeft = computed(() => computePath(-0.5))
const pathRight = computed(() => computePath(4.5))

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
  grid: { left: 50, right: 30, top: 30, bottom: 40 },
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
  legend: {
    data: ['f(x)', 'Start left (x₀=−0.5)', 'Start right (x₀=4.5)'],
    textStyle: { color: '#aaa', fontSize: 10 },
    top: 6,
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
      name: 'Start left (x₀=−0.5)',
      type: 'line',
      data: pathLeft.value,
      showSymbol: true,
      symbolSize: 7,
      lineStyle: { width: 1.5, color: '#f87171', type: 'dashed' },
      itemStyle: { color: '#f87171' },
      label: {
        show: true,
        formatter: (p) => p.dataIndex === 0 ? `x₀` : p.dataIndex === pathLeft.value.length - 1 ? `x${p.dataIndex}` : '',
        position: 'top',
        color: '#f87171',
        fontSize: 10,
      },
    },
    {
      name: 'Start right (x₀=4.5)',
      type: 'line',
      data: pathRight.value,
      showSymbol: true,
      symbolSize: 7,
      lineStyle: { width: 1.5, color: '#34d399', type: 'dashed' },
      itemStyle: { color: '#34d399' },
      label: {
        show: true,
        formatter: (p) => p.dataIndex === 0 ? `x₀` : p.dataIndex === pathRight.value.length - 1 ? `x${p.dataIndex}` : '',
        position: 'top',
        color: '#34d399',
        fontSize: 10,
      },
    },
  ],
}))
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
</template>
