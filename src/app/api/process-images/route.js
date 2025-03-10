import { NextResponse } from "next/server";

export async function POST(request) {
    try {
        // Parse the multipart form data
        const formData = await request.formData();
        const preImage = formData.get("preImage");
        const postImage = formData.get("postImage");

        if (!preImage || !postImage) {
            return NextResponse.json({ error: "Both pre and post images are required" }, { status: 400 });
        }

        // Create a new FormData object to send to Python API
        const pythonFormData = new FormData();
        pythonFormData.append("preImage", preImage);
        pythonFormData.append("postImage", postImage);

        // Call Python API
        const pythonApiUrl = process.env.PYTHON_API_URL || "http://localhost:5000";

        const response = await fetch(pythonApiUrl, {
            method: "POST",
            body: pythonFormData,
        });

        if (!response.ok) {
            throw new Error(`Python API responded with status: ${response.status}`);
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("Error processing images:", error);
        return NextResponse.json({ error: "Failed to process images" }, { status: 500 });
    }
}
