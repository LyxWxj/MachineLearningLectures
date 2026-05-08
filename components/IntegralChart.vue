<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  a: { type: Number, default: 0 },
  b: { type: Number, default: 2 },
})

function f(x) {
  return 0.5 * x * x + 0.3
}

const step = 0.02
const xMin = -0.5
const xMax = 3

const xVals = computed(() => {
  const vals = []
  for (let x = xMin; x <= xMax; x += step) vals.push(+x.toFixed(4))
  return vals
})

// Compute integral value using trapezoidal rule
const integralValue = computed(() => {
  let sum = 0
  const s = 0.001
  for (let x = props.a; x < props.b; x += s) {
    sum += (f(x) + f(x + s)) * s / 2
  }
  return sum.toFixed(3)
})

const option = computed(() => ({
  animation: false,
  grid: { left: 50, right: 30, top: 20, bottom: 40 },
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
    max: 3,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  series: [
    {
      type: 'line',
      data: xVals.value.map(x => [x, f(x)]),
      showSymbol: false,
      lineStyle: { width: 2, color: '#60a5fa' },
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(96, 165, 250, 0.4)' },
            { offset: 1, color: 'rgba(96, 165, 250, 0.05)' },
          ],
        },
        origin: 'auto',
      },
      markArea: {
        silent: true,
        data: [
          [
            { xAxis: props.a, itemStyle: { color: 'rgba(96, 165, 250, 0.25)' } },
            { xAxis: props.b },
          ],
        ],
      },
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: '#fbbf24', type: 'dashed', width: 1 },
        data: [
          { xAxis: props.a },
          { xAxis: props.b },
        ],
        label: {
          show: true,
          formatter: (p) => p.dataIndex === 0 ? `a=${props.a}` : `b=${props.b}`,
          color: '#fbbf24',
        },
      },
    },
  ],
}))
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
  <div class="text-center text-sm text-gray-400 mt-2">
    <span v-katex>`\int_{ {{a}} }^{ {{b}} } f(x)\,dx \approx {{ integralValue }}`</span>
  </div>
</template>
