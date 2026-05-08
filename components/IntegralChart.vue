<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { GridComponent, TooltipComponent, LegendComponent, MarkAreaComponent, MarkLineComponent } from 'echarts/components'

use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,        // 新增
  MarkAreaComponent,
  MarkLineComponent,     // 新增
  CanvasRenderer
])
const props = defineProps({
  a: { type: Number, default: 0.5 },
  b: { type: Number, default: 2 },
})

function f(x) {
  return 0.5 * x * x + 0.3
}

// Antiderivative F(x) = x³/6 + 0.3x
function F(x) {
  return (x ** 3) / 6 + 0.3 * x
}

const step = 0.02
const xMin = -0.5
const xMax = 3

const xVals = computed(() => {
  const vals = []
  for (let x = xMin; x <= xMax; x += step) vals.push(+x.toFixed(4))
  return vals
})

const integralValue = computed(() => (F(props.b) - F(props.a)).toFixed(3))

const option = computed(() => ({
  animation: false,
  grid: { left: 50, right: 30, top: 20, bottom: 40 },
  legend: {
    data: ['f(x)', 'F(x)'],
    top: 0,
    right: 10,
    textStyle: { color: '#aaa', fontSize: 11 },
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: '#222',
    borderColor: '#555',
    textStyle: { color: '#eee', fontSize: 12 },
    formatter: (params) => {
      const x = params[0].data[0]
      return `x=${x.toFixed(2)}<br/>f(x)=${f(x).toFixed(3)}<br/>F(x)=${F(x).toFixed(3)}`
    },
  },
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
    max: 4,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  series: [
    // f(x) curve
    {
      name: 'f(x)',
      type: 'line',
      data: xVals.value.map(x => [x, f(x)]),
      showSymbol: false,
      lineStyle: { width: 2, color: '#60a5fa' },
      smooth: true,
      z: 2,
    },
    // F(x) accumulation curve
    {
      name: 'F(x)',
      type: 'line',
      data: xVals.value.map(x => [x, F(x)]),
      showSymbol: false,
      lineStyle: { width: 2, color: '#a78bfa' },
      smooth: true,
      z: 2,
      markArea: {
        silent: true,
        data: [
          [
            { xAxis: props.a, itemStyle: { color: 'rgba(167, 139, 250, 0.15)' } },
            { xAxis: props.b },
          ],
        ],
      },
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: '#a78bfa', type: 'dashed', width: 1 },
        data: [
          { xAxis: props.a },
          { xAxis: props.b },
        ],
        label: {
          show: true,
          formatter: (p) => p.dataIndex === 0 ? `F(a)=${F(props.a).toFixed(2)}` : `F(b)=${F(props.b).toFixed(2)}`,
          color: '#a78bfa',
        },
      },
    },
  ],
}))
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
  <div class="text-center text-sm text-gray-400 mt-2">
    <span v-katex>`\int_a^b f(x)\,dx = F(b) - F(a) = {{ integralValue }}`</span>
  </div>
</template>
