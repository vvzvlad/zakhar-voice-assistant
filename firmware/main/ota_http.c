#include "ota_http.h"

#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "ota";

#define REBOOT_DELAY_MS 500  // lets the reply leave before the reboot

// The server task runs one handler at a time, so one buffer serves every upload.
static char s_buf[4096];

static esp_err_t fail(httpd_req_t *req, const char *what, esp_err_t err)
{
    ESP_LOGE(TAG, "%s: %s", what, esp_err_to_name(err));
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, what);
    return ESP_FAIL;
}

static esp_err_t ota_post(httpd_req_t *req)
{
    // An image that takes an upload is good enough to keep; and while it is unconfirmed,
    // esp_ota_begin refuses (ESP_ERR_OTA_ROLLBACK_INVALID_STATE).
    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(esp_ota_get_running_partition(), &state) == ESP_OK &&
        state == ESP_OTA_IMG_PENDING_VERIFY) {
        esp_ota_mark_app_valid_cancel_rollback();
    }
    const esp_partition_t *part = esp_ota_get_next_update_partition(NULL);
    ESP_LOGI(TAG, "receiving %u B into %s", (unsigned)req->content_len, part->label);
    esp_ota_handle_t ota;
    esp_err_t err = esp_ota_begin(part, OTA_WITH_SEQUENTIAL_WRITES, &ota);
    if (err != ESP_OK) return fail(req, "ota begin", err);
    size_t left = req->content_len;
    while (left > 0) {
        const int n = httpd_req_recv(req, s_buf, left < sizeof s_buf ? left : sizeof s_buf);
        if (n <= 0) {
            esp_ota_abort(ota);
            return fail(req, "receive", ESP_FAIL);
        }
        err = esp_ota_write(ota, s_buf, (size_t)n);
        if (err != ESP_OK) {
            esp_ota_abort(ota);
            return fail(req, "write", err);
        }
        left -= (size_t)n;
    }
    err = esp_ota_end(ota);  // validates the image
    if (err != ESP_OK) return fail(req, "image", err);
    err = esp_ota_set_boot_partition(part);
    if (err != ESP_OK) return fail(req, "boot partition", err);
    httpd_resp_sendstr(req, "OK, rebooting\n");
    ESP_LOGI(TAG, "%s is next, rebooting", part->label);
    vTaskDelay(pdMS_TO_TICKS(REBOOT_DELAY_MS));
    esp_restart();
    return ESP_OK;
}

void ota_http_start(void)
{
    const httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server;
    esp_err_t err = httpd_start(&server, &cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "server start failed: %s", esp_err_to_name(err));
        return;
    }
    const httpd_uri_t uri = {.uri = "/ota", .method = HTTP_POST, .handler = ota_post};
    httpd_register_uri_handler(server, &uri);
    ESP_LOGI(TAG, "POST /ota on port %u", (unsigned)cfg.server_port);
}
