import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Helper to find a node by ID, handling both numeric and string IDs.
 * ComfyUI may use UUIDs when LiteGraph.use_uuids is active.
 */
function getNodeById(nodeId) {
    const graph = app.graph;
    if (!graph) return null;
    return graph.getNodeById(nodeId) || graph.getNodeById(+nodeId) || null;
}

app.registerExtension({
    name: "import_itera.batch_image_saver",

    async setup() {
        api.addEventListener("batch_saver.update", (event) => {
            const data = event.detail;
            const nodeId = data.node;
            if (nodeId == null) return;

            const node = getNodeById(nodeId);
            if (!node) return;

            const infoWidget = node.widgets?.find((w) => w.name === "save_info");
            if (infoWidget) {
                infoWidget.value = data.info || "";
            }

            node.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "BatchImageSaver") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Custom draw widget for save info display
            const infoWidget = {
                type: "custom",
                name: "save_info",
                value: "Awaiting first save...",
                serialize: false,
                computeSize() {
                    const lines = (this.value || "").split("\n").length;
                    return [0, Math.max(30, lines * 16 + 10)];
                },
                draw(ctx, node, widgetWidth, y) {
                    const lines = (this.value || "").split("\n");
                    ctx.save();
                    ctx.font = "11px monospace";
                    ctx.fillStyle = "#88cc88";
                    ctx.textAlign = "left";

                    const margin = 15;
                    for (let i = 0; i < lines.length; i++) {
                        ctx.fillText(lines[i], margin, y + 14 + i * 15);
                    }
                    ctx.restore();
                },
            };
            this.addCustomWidget(infoWidget);

            // Adjust node size
            const w = Math.max(this.size[0], 340);
            this.setSize([w, this.size[1] + 60]);

            return result;
        };

        // Handle onExecuted for UI text fallback
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message?.text) {
                const infoWidget = this.widgets?.find(
                    (w) => w.name === "save_info"
                );
                if (infoWidget) {
                    infoWidget.value = Array.isArray(message.text)
                        ? message.text.join("\n")
                        : String(message.text);
                }
                this.setDirtyCanvas(true, true);
            }
        };
    },
});
