<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, CanvasRenderer])

const props = defineProps({
  xMin: { type: Number, default: -2 },
  xMax: { type: Number, default: 2.5 },
  tangentX: { type: Number, default: 1 },
})

function f(x) {
  return x ** 3 / 3 - x
}

function fPrime(x) {
  return x ** 2 - 1
}

const step = 0.05
const xVals = computed(() => {
  const vals = []
  for (let x = props.xMin; x <= props.xMax; x += step) vals.push(+x.toFixed(4))
  return vals
})
const yVals = computed(() => xVals.value.map(f))

const activeX = ref(props.tangentX)
const tangentY = computed(() => f(activeX.value))
const slope = computed(() => fPrime(activeX.value))

const tangentLineData = computed(() => {
  const x0 = activeX.value, y0 = tangentY.value, m = slope.value, dx = 0.8
  return [[x0 - dx, y0 - m * dx], [x0 + dx, y0 + m * dx]]
})

const option = computed(() => ({
  animation: false,
  grid: { left: 50, right: 30, top: 20, bottom: 40 },
  xAxis: {
    type: 'value', min: props.xMin, max: props.xMax,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  yAxis: {
    type: 'value', min: -1.5, max: 1.5,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
  },
  series: [
    {
      type: 'line',
      data: xVals.value.map((x, i) => [x, yVals.value[i]]),
      showSymbol: false,
      lineStyle: { width: 2, color: '#60a5fa' },
      smooth: true,
    },
    {
      type: 'line',
      data: tangentLineData.value,
      showSymbol: false,
      lineStyle: { width: 2, color: '#f87171', type: 'dashed' },
      z: 10,
    },
    {
      type: 'scatter',
      data: [[activeX.value, tangentY.value]],
      symbolSize: 14,
      cursor: 'grab',
      itemStyle: { color: '#f87171' },
      label: {
        show: true,
        formatter: `x=${activeX.value.toFixed(2)}, f'=${slope.value.toFixed(2)}`,
        position: 'top', color: '#f87171', fontSize: 12,
      },
      z: 10,
    },
  ],
}))

// Drag interaction
const chartRef = ref(null)
const dragging = ref(false)

function getChart() {
  return chartRef.value?.chart?.value || chartRef.value?.chart
}

function updateXFromPixel(pixelX) {
  const chart = getChart()
  if (!chart) return
  try {
    const point = chart.convertFromPixel({ seriesIndex: 0 }, [pixelX, 0])
    if (point) {
      activeX.value = Math.max(props.xMin, Math.min(props.xMax, +point[0].toFixed(4)))
    }
  } catch {}
}

function onZrMouseDown(e) {
  const chart = getChart()
  if (!chart) return
  const pixel = [e.offsetX, e.offsetY]
  const dataPoint = chart.convertFromPixel({ seriesIndex: 2 }, pixel)
  if (!dataPoint) return
  const dx = Math.abs(dataPoint[0] - activeX.value)
  const dy = Math.abs(dataPoint[1] - tangentY.value)
  // Hit test: within ~0.3 units in x and ~0.5 units in y
  if (dx < 0.3 && dy < 0.5) {
    dragging.value = true
    chartRef.value.root.style.cursor = 'grabbing'
  }
}

function onZrMouseMove(e) {
  if (!dragging.value) return
  updateXFromPixel(e.offsetX)
}

function onZrMouseUp() {
  if (!dragging.value) return
  dragging.value = false
  const root = chartRef.value?.root
  if (root) root.style.cursor = ''
}

onMounted(() => {
  const zr = getChart()?.getZr()
  if (zr) {
    zr.on('mousedown', onZrMouseDown)
    zr.on('mousemove', onZrMouseMove)
    zr.on('mouseup', onZrMouseUp)
  }
})

onBeforeUnmount(() => {
  const zr = getChart()?.getZr()
  if (zr) {
    zr.off('mousedown', onZrMouseDown)
    zr.off('mousemove', onZrMouseMove)
    zr.off('mouseup', onZrMouseUp)
  }
})
</script>

<template>
  <VChart ref="chartRef" :option="option" autoresize style="width: 100%; height: 100%;" />
</template>
