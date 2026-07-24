# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Với temperature 0.0, mô hình trả lời rất chính xác và mạch lạc, cung cấp một sự thật lịch sử cụ thể về tuyến tàu điện leng keng từ năm 1901. Khi temperature tăng lên 0.7 và 1.2, câu trả lời vẫn giữ được sự mạch lạc nhưng thêm vào các chi tiết văn hóa và lịch sử phong phú hơn như "phố cổ", "tầng lớp quý tộc", "điểm hẹn", tạo nên một bức tranh sống động. Tuy nhiên, ở temperature 1.8, mô hình bắt đầu có dấu hiệu kém mạch lạc khi đưa ra thông tin không liên quan trực tiếp đến Hà Nội mà lại nói về ẩm thực Việt Nam nói chung ("phở, bún chả") và kết thúc câu một cách đột ngột, không logic, cho thấy sự mất kiểm soát về nội dung.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Để soạn thảo hợp đồng pháp lý, tôi sẽ đặt temperature ở mức 0.2–0.3. Mức này đủ thấp để đảm bảo tính nhất quán, chính xác và tuân thủ chặt chẽ các quy định pháp luật, tránh các biến thể không mong muốn hoặc sáng tạo quá mức có thể dẫn đến rủi ro pháp lý. Ngược lại, để viết slogan quảng cáo, tôi sẽ đặt temperature ở mức 0.8–1.0. Mức này khuyến khích sự sáng tạo, độc đáo và bất ngờ, giúp tạo ra các khẩu hiệu ấn tượng, dễ nhớ và phù hợp với xu hướng thị trường. Sự khác biệt nằm ở mục tiêu của từng ứng dụng: hợp đồng đòi hỏi sự an toàn và chính xác tuyệt đối, trong khi quảng cáo ưu tiên sự mới mẻ và thu hút.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Với 20.000 người dùng, mỗi người gọi 2 lần/ngày, mỗi lần 500 token đầu ra, tổng lượng token đầu ra mỗi ngày là 20.000 × 2 × 500 = 20.000.000 token.
Model lớn (gpt-4o): 20.000.000 × $0.01 / 1.000 = $200 mỗi ngày.
Model nhỏ (gpt-4o-mini): 20.000.000 × $0.0006 / 1.000 = $12 mỗi ngày.
Trường hợp model lớn xứng đáng là trợ lý nghiên cứu khoa học, cần phân tích tài liệu phức tạp, đòi hỏi suy luận đa bước và kiến thức chuyên sâu.
Trường hợp model nhỏ là chatbot hỗ trợ khách hàng đơn giản, chỉ trả lời các câu hỏi thường gặp và không cần xử lý logic phức tạp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona nhà thơ, phản hồi mang văn phong bay bổng, sử dụng nhiều hình ảnh ví von sinh động và hoàn toàn không chứa thuật ngữ kỹ thuật. Ngược lại, với persona kỹ sư senior, phản hồi mang tính cấu trúc mạch lạc, chính xác, tập trung vào mô hình dữ liệu và đi kèm ví dụ code Python cụ thể. Điều này chứng minh rằng system prompt có khả năng điều khiển trực tiếp giọng văn (tone of voice), mức độ kỹ thuật (đơn giản hay chuyên sâu), định dạng đầu ra (code, thơ, bảng biểu) và phạm vi thuật ngữ được phép sử dụng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với đoạn văn tiếng Việt 150 từ, ước lượng thô (150 / 0.75 = 200 token) sẽ thấp hơn nhiều so với thực tế đếm bằng `tiktoken` (thường khoảng 260–300 token), hai con số chênh lệch nhau khoảng 30%–40%. Nếu dùng ước lượng thô (`số từ / 0.75`) để dự toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ bị **dự toán thiếu** ngân sách nghiêm trọng. Nguyên nhân là do các bộ tách từ (tokenizer) BPE của OpenAI được tối ưu cho tiếng Anh, nên các từ tiếng Việt mang dấu thanh và mã mã hóa Unicode đa byte sẽ bị tách thành nhiều subword/byte token hơn so với từ tiếng Anh tương đương.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming vì giúp giảm tối đa độ trễ cảm nhận (perceived latency), người dùng thấy câu trả lời xuất hiện ngay lập tức từng từ mà không phải chờ đợi toàn bộ câu phản hồi sinh xong. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm (c) hoàn toàn không cần streaming vì đây là tác vụ xử lý hàng loạt (batch process) không có tương tác người dùng thời gian thực, nhận toàn bộ kết quả một lần giúp đơn giản hóa việc lưu trữ dữ liệu và xử lý file. Riêng với trợ lý giọng nói (b), streaming có thể được tận dụng để đưa văn bản vào bộ tổng hợp giọng nói (TTS) sớm hơn, nhưng người dùng chỉ nghe thấy âm thanh khi từng cụm từ hoàn chỉnh được phát ra.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff giúp giảm áp lực dồn dập lên server bị quá tải bằng cách tự động giãn rộng khoảng thời gian giữa các lần thử lại sau mỗi lần thất bại, cho phép server có thời gian hồi phục thay vì bị tấn công từ chối dịch vụ vô tình (thảm họa thundering herd) khi dùng delay cố định. Tuy nhiên, nếu hàng nghìn client cùng gặp lỗi tại cùng một thời điểm và dùng cùng công thức backoff, chúng vẫn sẽ gửi yêu cầu thử lại đồng loạt tại các mốc thời gian trùng nhau. Kỹ thuật "jitter" (thêm độ trễ ngẫu nhiên vào khoảng thời gian chờ) giải quyết triệt để vấn đề này bằng cách làm phân tán các yêu cầu thử lại rải rác theo thời gian, triệt tiêu hiện tượng đỉnh xung đột yêu cầu trùng lặp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ giảng AI thân thiện của khóa học lập trình. Hãy trả lời ngắn gọn dưới 3 câu và luôn đi kèm một ví dụ code nhỏ nếu câu hỏi liên quan đến lập trình."
> Hai vị trí quan trọng nếu xóa đi:
> 1. Xóa "ngắn gọn dưới 3 câu": Trợ lý sẽ mặc định giải thích rất dài dòng, liệt kê lý thuyết chi tiết gây tốn kém token và làm tăng độ trễ phản hồi.
> 2. Xóa "luôn đi kèm một ví dụ code nhỏ nếu câu hỏi liên quan đến lập trình": Trợ lý sẽ chỉ trả lời bằng văn bản thuần túy, khiến người học khó hình dung trực quan cách áp dụng mã nguồn thực tế.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: Ở lượt 1, người dùng khai báo "Tôi đang viết ứng dụng backend bằng FastAPI và PostgreSQL". Sau 5 lượt trao đổi về các câu hỏi nhỏ khác, đến lượt 6 người dùng hỏi "Làm sao để tạo migration cho database của tôi?". Do history chỉ lưu 4 lượt gần nhất, thông tin về "FastAPI và PostgreSQL" ở lượt 1 đã bị loại bỏ khỏi ngữ cảnh, khiến trợ lý không biết khung làm việc người dùng đang sử dụng và có thể hướng dẫn nhầm sang Django hoặc Alembic/SQLAlchemy chung chung.
> Cách khắc phục: Triển khai kỹ thuật tóm tắt ngữ cảnh (Context Summarization) — mỗi khi lịch sử hội thoại vượt quá 4 lượt, sử dụng một LLM chạy ngầm tóm tắt các lượt cũ thành một đoạn tóm tắt ngắn (System Memory) và chèn vào đầu ngữ cảnh hội thoại thay vì xóa bỏ hoàn toàn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [✓] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [✓] Cả 4 checkpoint pytest đều pass
- [✓] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
