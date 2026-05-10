<template>
  <div class="flex flex-col items-center">
    <div class="mb-2 text-sm text-gray-400">
      Shape: {{ rows }} × {{ cols }}
    </div>
    <svg :width="svgWidth" :height="svgHeight">
      <g v-for="(row, i) in grid" :key="i">
        <rect
          v-for="(cell, j) in row"
          :key="j"
          :x="j * cellSize"
          :y="i * cellSize"
          :width="cellSize"
          :height="cellSize"
          :fill="getCellColor(cell)"
          stroke="#4a5568"
          stroke-width="1"
        />
        <text
          v-for="(cell, j) in row"
          :key="'text-' + j"
          :x="j * cellSize + cellSize / 2"
          :y="i * cellSize + cellSize / 2"
          text-anchor="middle"
          dominant-baseline="central"
          :fill="cell > 128 ? '#000' : '#fff'"
          :font-size="cellSize * 0.35"
        >
          {{ cell }}
        </text>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps({
  rows: { type: Number, default: 4 },
  cols: { type: Number, default: 4 },
  cellSize: { type: Number, default: 50 },
  data: { type: Array as () => number[][], default: null }
})

const grid = computed(() => {
  if (props.data) return props.data
  // Generate random grayscale values
  return Array.from({ length: props.rows }, () =>
    Array.from({ length: props.cols }, () => Math.floor(Math.random() * 256))
  )
})

const svgWidth = computed(() => props.cols * props.cellSize)
const svgHeight = computed(() => props.rows * props.cellSize)

function getCellColor(value) {
  return `rgb(${value}, ${value}, ${value})`
}
</script>
