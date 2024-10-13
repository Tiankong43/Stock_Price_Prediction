<template>
  <div style="display: flex;
  align-items: center;
  justify-content: center;">
    <div ref="chart" class="chart"></div>
  </div>
</template>

<script setup>
import axios from 'axios'
import {ref, onMounted} from "vue";
import * as echarts from 'echarts/core';
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  VisualMapComponent
} from 'echarts/components';
import {LineChart} from 'echarts/charts';
import {UniversalTransition} from 'echarts/features';
import {CanvasRenderer} from 'echarts/renderers';
import { LegendComponent } from 'echarts/components';echarts.use([LegendComponent]);


echarts.use([
  TitleComponent,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  LineChart,
  CanvasRenderer,
  UniversalTransition
]);
const chart = ref()
onMounted(async () => {
  const data = [];
  const res = await axios.get("http://127.0.0.1:5000/getdata")
  res.data.map(x=>{
    data.push(x)
  })
  var myChart = echarts.init(chart.value);
  var option;

// prettier-ignore

const date = data.map(function (item) {
  return item[0];
});

const value = data.map(function (item) {
  return item[1];
});

option = {
  legend: {
    data: ['实际数据','预测数据'],
    orient: 'vertical',
    bottom: 'middle',
    itemGap: 20,
    top: '0%',
    right: '0%'
  },
  tooltip: {
    trigger: 'axis'
  },
  title: {
      left: 'center',
      text: '历史20天及未来10天股票价格'
    },
  xAxis: {
    type: 'category',
    data: date,
    name: '日期'
  },
  yAxis: {
  type: 'value',
  name: '股票数据',
  splitNumber: 5,
  splitLine: {
    show: true,
  },
  min: 'dataMin',
  max: 'dataMax',
  boundaryGap: ['20%', '20%'],
},
visualMap: {
            type: "piecewise",
            show: false,
            dimension: 0,
            seriesIndex: 0,
            pieces: [
                {
                    gt: 0,
                    lt: 20,
                    color: "#00aaff",
                },
                {
                    gt: 20,
                    color: "#ff0000",
                },
            ],
        },
  series: [
    {
      data: value,
      type: 'line',
      label: {
        show: true,
        position: 'top',
        textStyle: {
          fontSize: 12
        }
      }
    },
    {
      name: '实际数据',
      type: 'line',
      showSymbol: false,
      hoverAnimation: false,
      color: '#00aaff',
      data: [],
    },
    {
      name: '预测数据',
      type: 'line',
      showSymbol: false,
      hoverAnimation: false,
      color: '#ff0000',
      data: [],
    }
  ]
};

option && myChart.setOption(option);
})

</script>


<style scoped>
.chart {
  width: 1080px;
  height: 750px;
}
</style>


