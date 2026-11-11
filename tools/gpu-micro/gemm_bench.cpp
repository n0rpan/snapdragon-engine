// moe-prefill GPU feasibility micro benchmark: GEMMs at prefill shapes through a minimal Vulkan compute
// shader (gemm.comp), no model. See the GPU prefill study in the private lab notes. Derived from the lab's earlier
// gpu-investigation/dot_bench.cpp. Build: g++ -O2 -std=c++17 gemm_bench.cpp -lvulkan -o gemm_bench
// Run:   ./gemm_bench <dir with gemm-N.spv>     (a few hundred ms of GPU time in total)
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(VkResult r, const char * s) { if (r != VK_SUCCESS) throw std::runtime_error(std::string(s) + ":" + std::to_string(r)); }
#define VKC(x) check(x, #x)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// IEEE half from float (round to nearest even), enough for values in (-1, 1)
static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int e = int((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (e <= 0) { if (e < -10) return uint16_t(sign); m |= 0x800000u; uint32_t sh = uint32_t(14 - e); uint32_t h = m >> sh; if ((m >> (sh - 1)) & 1u) h++; return uint16_t(sign | h); }
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    uint32_t h = (uint32_t(e) << 10) | (m >> 13); if (m & 0x1000u) h++; return uint16_t(sign | h);
}
static float h2f(uint16_t h) {
    int e = (h >> 10) & 31; uint32_t m = h & 0x3ffu; float v;
    if (e == 0) v = std::ldexp(float(m), -24); else v = std::ldexp(float(m | 0x400u), e - 25);
    return (h & 0x8000u) ? -v : v;
}
struct Buf { VkBuffer b; VkDeviceMemory m; void * p; VkDeviceSize n; };
int main(int argc, char ** argv) { try {
    const std::string root = argc > 1 ? argv[1] : ".";
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName = "moepf-gemm-micro"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app; VkInstance instance; VKC(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t nd = 0; VKC(vkEnumeratePhysicalDevices(instance, &nd, nullptr)); if (!nd) throw std::runtime_error("no device");
    std::vector<VkPhysicalDevice> devs(nd); VKC(vkEnumeratePhysicalDevices(instance, &nd, devs.data())); VkPhysicalDevice pd = devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device=%s api=%u.%u.%u timestampPeriod=%f\n", props.deviceName, VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion), props.limits.timestampPeriod);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr); std::vector<VkQueueFamilyProperties> qp(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp.data());
    uint32_t qi = 0; while (qi < nq && !(qp[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++; if (qi == nq) throw std::runtime_error("no compute queue");
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.shaderFloat16 = VK_TRUE; f12.pNext = &f13;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.storageBuffer16BitAccess = VK_TRUE; f11.pNext = &f12;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext = &f11; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; VKC(vkCreateDevice(pd, &dci, nullptr, &dev)); VkQueue queue; vkGetDeviceQueue(dev, qi, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto mkbuf = [&](VkDeviceSize n) {
        Buf b{}; b.n = n; VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = n; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VKC(vkCreateBuffer(dev, &bci, nullptr, &b.b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, b.b, &req);
        // unified memory: host-visible + coherent + CACHED, so the CPU reads results at normal speed (zero copy)
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        uint32_t mt = 0; for (; mt < mp.memoryTypeCount; mt++) if ((req.memoryTypeBits & (1u << mt)) && (mp.memoryTypes[mt].propertyFlags & want) == want) break;
        if (mt == mp.memoryTypeCount) throw std::runtime_error("no cached coherent memory type");
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
        VKC(vkAllocateMemory(dev, &mai, nullptr, &b.m)); VKC(vkBindBufferMemory(dev, b.b, b.m, 0)); VKC(vkMapMemory(dev, b.m, 0, n, 0, &b.p)); return b; };
    VkDescriptorSetLayoutBinding bind[3]{}; for (int i = 0; i < 3; i++) { bind[i].binding = i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; slci.bindingCount = 3; slci.pBindings = bind; VkDescriptorSetLayout sl; VKC(vkCreateDescriptorSetLayout(dev, &slci, nullptr, &sl));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16}; VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount = 1; plci.pSetLayouts = &sl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &push;
    VkPipelineLayout layout; VKC(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}; VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool pool; VKC(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool)); VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
    VkDescriptorSet ds; VKC(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpci.queueFamilyIndex = qi; VkCommandPool cp; VKC(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1; VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2; VkQueryPool qpool; VKC(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VKC(vkCreateFence(dev, &fci, nullptr, &fence));
    struct Case { int mode; const char * name; uint32_t R, K, T, iters; };
    const Case cases[] = {
        {0, "int8 dot4x8 peak (registers only)", 512, 2048, 512, 1024},
        {1, "fp16 vec4 fma peak (registers only)", 512, 2048, 512, 1024},
        {2, "int8 GEMM 1 token/invocation", 512, 2048, 256, 0},
        {3, "int8 GEMM 4 tokens/invocation", 512, 2048, 256, 0},
        {3, "int8 GEMM 4 tokens/invocation (qkv)", 8192, 2048, 256, 0},
        {4, "fp16 GEMM 1 token/invocation", 512, 2048, 256, 0},
        {5, "fp16 GEMM 4 tokens/invocation", 512, 2048, 256, 0},
        {5, "fp16 GEMM 4 tokens/invocation (qkv)", 8192, 2048, 256, 0},
        {5, "fp16 GEMM 4 tokens, 24-token ubatch", 8192, 2048, 24, 0},
    };
    std::printf("%-40s %6s %6s %5s %10s %10s %10s %9s\n", "case", "R", "K", "T", "gpu_ms", "wall_ms", "GMAC/s", "check");
    for (const Case & c : cases) {
        const bool f16 = c.mode == 4 || c.mode == 5; const VkDeviceSize esz = f16 ? 2 : 1;
        Buf W = mkbuf(VkDeviceSize(c.R) * c.K * esz), X = mkbuf(VkDeviceSize(c.T) * c.K * esz), O = mkbuf(VkDeviceSize(c.R) * c.T * 4 + 64);
        uint32_t seed = 7; auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
        if (f16) { uint16_t * w = (uint16_t *) W.p, * x = (uint16_t *) X.p;
            for (VkDeviceSize i = 0; i < VkDeviceSize(c.R) * c.K; i++) w[i] = f2h(float(int(rnd() >> 16 & 0xff) - 128) / 1024.0f);
            for (VkDeviceSize i = 0; i < VkDeviceSize(c.T) * c.K; i++) x[i] = f2h(float(int(rnd() >> 16 & 0xff) - 128) / 128.0f);
        } else { uint8_t * w = (uint8_t *) W.p, * x = (uint8_t *) X.p;
            for (VkDeviceSize i = 0; i < W.n; i++) w[i] = uint8_t(rnd() >> 16);
            for (VkDeviceSize i = 0; i < X.n; i++) x[i] = uint8_t(rnd() >> 16); }
        VkDescriptorBufferInfo dbi[3] = {{W.b, 0, W.n}, {X.b, 0, X.n}, {O.b, 0, O.n}}; VkWriteDescriptorSet wr[3]{};
        for (int i = 0; i < 3; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i]; }
        vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);
        std::ifstream f(root + "/gemm-" + std::to_string(c.mode) + ".spv", std::ios::binary | std::ios::ate); if (!f) throw std::runtime_error("missing shader");
        size_t len = f.tellg(); std::vector<uint32_t> code(len / 4); f.seekg(0); f.read((char *) code.data(), len);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize = len; smci.pCode = code.data(); VkShaderModule sm; VKC(vkCreateShaderModule(dev, &smci, nullptr, &sm));
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = sm; pci.stage.pName = "main"; pci.layout = layout;
        VkPipeline pipe; VKC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe));
        const bool tok4 = c.mode == 3 || c.mode == 5; const uint64_t inv = (c.mode <= 1) ? uint64_t(c.R) * c.T : uint64_t(c.R) * (tok4 ? (c.T + 3) / 4 : c.T);
        std::vector<double> tg, tw;
        for (int rep = 0; rep < 6; rep++) {
            VKC(vkResetCommandBuffer(cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VKC(vkBeginCommandBuffer(cb, &bi));
            vkCmdResetQueryPool(cb, qpool, 0, 2); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
            uint32_t pc[4] = {c.R, c.K, c.T, c.iters}; vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
            vkCmdDispatch(cb, uint32_t((inv + 63) / 64), 1, 1); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1); VKC(vkEndCommandBuffer(cb));
            VKC(vkResetFences(dev, 1, &fence)); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            const double t0 = now(); VKC(vkQueueSubmit(queue, 1, &si, fence)); VKC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30000000000ull)); const double t1 = now();
            uint64_t ts[2]; VKC(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            if (rep) { tg.push_back(double(ts[1] - ts[0]) * props.limits.timestampPeriod / 1e6); tw.push_back((t1 - t0) * 1e3); }
        }
        std::sort(tg.begin(), tg.end()); std::sort(tw.begin(), tw.end());
        const double mac = (c.mode <= 1) ? double(inv) * c.iters * 4.0 : double(c.R) * c.K * c.T;
        // check a sample of outputs against the CPU
        uint32_t bad = 0, n = 0; double maxrel = 0;
        if (c.mode >= 2) for (uint32_t s = 0; s < 64; s++) { uint32_t t = (s * 37u) % c.T, r = (s * 101u) % c.R;
            if (f16) { const uint16_t * w = (const uint16_t *) W.p + size_t(r) * c.K, * x = (const uint16_t *) X.p + size_t(t) * c.K; double a = 0, sc = 0;
                for (uint32_t k = 0; k < c.K; k++) { a += double(h2f(w[k])) * h2f(x[k]); sc += std::fabs(double(h2f(w[k])) * h2f(x[k])); }
                const double g = ((const float *) O.p)[size_t(t) * c.R + r]; maxrel = std::max(maxrel, std::fabs(g - a) / (sc + 1e-9)); if (std::fabs(g - a) > 1e-3 * sc + 1e-4) bad++;
            } else { const int8_t * w = (const int8_t *) W.p + size_t(r) * c.K, * x = (const int8_t *) X.p + size_t(t) * c.K; int32_t a = 0;
                for (uint32_t k = 0; k < c.K; k++) a += int32_t(w[k]) * x[k]; if (((const int32_t *) O.p)[size_t(t) * c.R + r] != a) bad++; }
            n++; }
        char chk[64]; if (c.mode < 2) std::snprintf(chk, sizeof chk, "-"); else if (f16) std::snprintf(chk, sizeof chk, "%u/%u bad, rel %.1e", bad, n, maxrel); else std::snprintf(chk, sizeof chk, "%u/%u bad", bad, n);
        std::printf("%-40s %6u %6u %5u %10.3f %10.3f %10.1f  %s\n", c.name, c.R, c.K, c.T, tg[2], tw[2], mac / (tg[2] * 1e-3) / 1e9, chk); std::fflush(stdout);
        vkDestroyPipeline(dev, pipe, nullptr); vkDestroyShaderModule(dev, sm, nullptr);
        for (Buf * b : {&W, &X, &O}) { vkUnmapMemory(dev, b->m); vkDestroyBuffer(dev, b->b, nullptr); vkFreeMemory(dev, b->m, nullptr); }
    }
    // sync cost: an (almost) empty dispatch, submit -> fence
    { Buf W = mkbuf(4096), X = mkbuf(4096), O = mkbuf(4096); VkDescriptorBufferInfo dbi[3] = {{W.b, 0, W.n}, {X.b, 0, X.n}, {O.b, 0, O.n}}; VkWriteDescriptorSet wr[3]{};
      for (int i = 0; i < 3; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i]; }
      vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);
      std::ifstream f(root + "/gemm-2.spv", std::ios::binary | std::ios::ate); size_t len = f.tellg(); std::vector<uint32_t> code(len / 4); f.seekg(0); f.read((char *) code.data(), len);
      VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize = len; smci.pCode = code.data(); VkShaderModule sm; VKC(vkCreateShaderModule(dev, &smci, nullptr, &sm));
      VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = sm; pci.stage.pName = "main"; pci.layout = layout;
      VkPipeline pipe; VKC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe)); std::vector<double> tw;
      for (int rep = 0; rep < 40; rep++) { VKC(vkResetCommandBuffer(cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VKC(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
        uint32_t pc[4] = {4, 16, 1, 0}; vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc); vkCmdDispatch(cb, 1, 1, 1); VKC(vkEndCommandBuffer(cb));
        VKC(vkResetFences(dev, 1, &fence)); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        const double t0 = now(); VKC(vkQueueSubmit(queue, 1, &si, fence)); VKC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30000000000ull)); if (rep >= 8) tw.push_back((now() - t0) * 1e6); }
      std::sort(tw.begin(), tw.end()); std::printf("sync: empty dispatch submit->fence  median %.0f us  min %.0f  p90 %.0f\n", tw[tw.size() / 2], tw[0], tw[tw.size() * 9 / 10]);
      vkDestroyPipeline(dev, pipe, nullptr); vkDestroyShaderModule(dev, sm, nullptr); }
    VKC(vkDeviceWaitIdle(dev)); return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; } }
