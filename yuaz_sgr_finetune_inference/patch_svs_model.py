diff --git a/models/svs_model.py b/models/svs_model.py
--- a/models/svs_model.py
+++ b/models/svs_model.py
@@ -111,6 +111,27 @@ class SVSModel(nn.Module):
                list(self.temporal_predictor.parameters()) + \
                list(self.f0_predictor.parameters())

+    def freeze_decoder_params(self):
+        """返回不包含解码器的生成器参数（冻结解码器时使用）"""
+        return list(self.encoder.parameters()) + \
+               list(self.rvq.parameters()) + \
+               list(self.flow_gen.parameters()) + \
+               list(self.cond_encoder.parameters()) + \
+               list(self.temporal_predictor.parameters()) + \
+               list(self.f0_predictor.parameters())
+
+    def freeze_decoder(self):
+        """冻结 DDSP 解码器所有参数"""
+        for param in self.ddsp_decoder.parameters():
+            param.requires_grad = False
+        print(f"  [冻结] DDSP 解码器: {sum(p.numel() for p in self.ddsp_decoder.parameters())/1e6:.1f}M 参数已冻结")
+
+    def unfreeze_decoder(self):
+        """解冻 DDSP 解码器"""
+        for param in self.ddsp_decoder.parameters():
+            param.requires_grad = True
+
     def discriminator_params(self):
         """返回判别器参数"""
         return list(self.mpd.parameters()) + list(self.msd.parameters())
