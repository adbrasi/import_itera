import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getNodeById(nodeId) {
    const graph = app.graph;
    if (!graph) return null;
    return graph.getNodeById(nodeId) || graph.getNodeById(+nodeId) || null;
}

app.registerExtension({
    name: "import_itera.batch_image_loader",

    async setup() {
        api.addEventListener("batch_loader.update", (event) => {
            const data = event.detail;
            const nodeId = data.node;
            if (nodeId == null) return;

            const node = getNodeById(nodeId);
            if (!node) return;

            const infoWidget = node.widgets?.find((w) => w.name === "info_display");
            if (infoWidget) {
                infoWidget.value = `${data.info}\n${data.next_info}`;
            }

            node.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "BatchImageLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const self = this;

            const infoWidget = {
                type: "custom",
                name: "info_display",
                value: "Awaiting first load...",
                serialize: false,
                computeSize() {
                    return [0, 44];
                },
                draw(ctx, node, widgetWidth, y) {
                    const lines = (this.value || "").split("\n");
                    ctx.save();
                    ctx.font = "12px monospace";
                    ctx.fillStyle = "#cccccc";
                    ctx.textAlign = "left";
                    const margin = 15;
                    for (let i = 0; i < lines.length; i++) {
                        ctx.fillText(lines[i], margin, y + 15 + i * 16);
                    }
                    ctx.restore();
                },
            };
            this.addCustomWidget(infoWidget);

            this.addWidget("button", "Reset Loader", "reset", () => {
                api.fetchApi("/batch_loader/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ node_id: String(self.id) }),
                })
                    .then((res) => res.json())
                    .then(() => {
                        const infoW = self.widgets?.find(
                            (w) => w.name === "info_display"
                        );
                        if (infoW) {
                            infoW.value = "Reset! Will re-scan on next run.";
                        }
                        self.setDirtyCanvas(true, true);
                    })
                    .catch((err) => {
                        console.error("Batch Loader reset error:", err);
                    });
            });

            const w = Math.max(this.size[0], 340);
            this.setSize([w, this.size[1] + 80]);

            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message?.text) {
                const infoWidget = this.widgets?.find(
                    (w) => w.name === "info_display"
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
