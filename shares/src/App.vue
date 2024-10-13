<template>
  <div class="container">
    <upFile></upFile>
    <div class="spacer"></div>
    <datashow></datashow>
  </div>
</template>

<script setup>
import upFile from "@/components/upFile"
import datashow from "@/components/data_show"
</script>

<style scoped>
.container {
}

.datashow {
  width: 120%; /* 设置宽度为父容器的80% */
  height: 50%; /* 设置高度为父容器的50% */
}

.spacer {
  margin-top: 50px;
}
</style>
<!-- ... -->
<datashow class="datashow"></datashow>