<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, CustomChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, CustomChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  a: { type: Number, default: 0.5 },
  b: { type: Number, default: 2 },
  n: { type: Number, default: 8 },
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

// Riemann sum value (left endpoint)
const riemannValue = computed(() => {
  const dx = (props.b - props.a) / props.n
  let sum = 0
  for (let i = 0; i < props.n; i++) {
    sum += f(props.a + i * dx) * dx
  }
  return sum.toFixed(3)
})

const option = computed(() => {
  const dx = (props.b - props.a) / props.n
  return {
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
      // Function curve
      {
        type: 'line',
        data: xVals.value.map(x => [x, f(x)]),
        showSymbol: false,
        lineStyle: { width: 2, color: '#60a5fa' },
        smooth: true,
        z: 2,
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
      // Riemann rectangles via custom series
      {
        type: 'custom',
        renderItem: (params, api) => {
          const children = []
          for (let i = 0; i < props.n; i++) {
            const xLeft = props.a + i * dx
            const height = f(xLeft)
            const p1 = api.coord([xLeft, 0])
            const p2 = api.coord([xLeft + dx, height])
            children.push({
              type: 'rect',
              shape: {
                x: p1[0],
                y: p2[1],
                width: api.size([dx, 0])[0],
                height: p1[1] - p2[1],
              },
              style: {
                fill: 'rgba(251, 191, 36, 0.25)',
                stroke: '#fbbf24',
                lineWidth: 1,
              },
            })
          }
          return { type: 'group', children }
        },
        data: [[props.a, 0]],
        z: 1,
      },
    ],
  }
})
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
  <div class="text-center text-sm text-gray-400 mt-2">
    <span v-katex>`Area Under Curve:{{ riemannValue }}`</span>
  </div>
</template>
