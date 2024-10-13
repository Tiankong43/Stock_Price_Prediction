<template>
  <div style="display:flex;justify-content: center;margin-top: 5%">
    <el-upload
        v-model:file-list="fileList"
        class="upload-demo"
        action="http://127.0.0.1:5000/upload"
        multiple
        :on-preview="handlePreview"
        :on-remove="handleRemove"
        :before-remove="beforeRemove"
        :on-success="uploadFileOK"
        :limit="3"
        :on-exceed="handleExceed"
    >
      <el-button type="primary">上传股票数据</el-button>
    </el-upload>
  </div>
</template>
<script setup>
import axios from 'axios';
import { ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
const input = ref('')
const fileList = ref([])

const handleRemove= (file, uploadFiles) => {
  console.log(file, uploadFiles)
}

const handlePreview= (uploadFile) => {
  console.log(uploadFile)
}
const uploadFileOK = ()=>{
  window.location.reload()
}

const handleExceed= (files, uploadFiles) => {
  ElMessage.warning(
      `The limit is 3, you selected ${files.length} files this time, add up to ${
          files.length + uploadFiles.length
      } totally`
  )
}

const beforeRemove= (uploadFile, uploadFiles) => {
  return ElMessageBox.confirm(
      `Cancel the transfer of ${uploadFile.name} ?`
  ).then(
      () => true,
      () => false
  )
}
</script>
