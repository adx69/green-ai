"use client";
import React, { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload";
import Image from "next/image";

export default function Dashboard() {
    const [preImage, setPreImage] = useState(null);
    const [postImage, setPostImage] = useState(null);
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handlePreImageUpload = (files) => {
        if (files && files.length > 0) {
            setPreImage(files[0]);
        }
    };

    const handlePostImageUpload = (files) => {
        if (files && files.length > 0) {
            setPostImage(files[0]);
        }
    };

    const processImages = async () => {
        if (!preImage || !postImage) {
            alert("Please upload both pre and post images");
            return;
        }

        setIsLoading(true);

        try {
            const formData = new FormData();
            formData.append("preImage", preImage);
            formData.append("postImage", postImage);

            const response = await fetch("/api/process-images", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to process images");
            }

            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error("Error processing images:", error);
            alert("Failed to process images. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-dvh w-full max-w-4xl mx-auto my-8 space-y-8">
            <h1 className="text-2xl font-bold text-center">Sapling Detection Analysis</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                    <h2 className="text-lg font-semibold">Pre-Image Upload</h2>
                    <div className="min-h-64 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg">
                        {preImage ? (
                            <div className="relative h-64 w-full">
                                <Image
                                    src={URL.createObjectURL(preImage)}
                                    alt="Pre-image preview"
                                    fill
                                    className="object-contain"
                                />
                            </div>
                        ) : (
                            <FileUpload onChange={handlePreImageUpload} />
                        )}
                    </div>
                </div>

                <div className="space-y-4">
                    <h2 className="text-lg font-semibold">Post-Image Upload</h2>
                    <div className="min-h-64 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg">
                        {postImage ? (
                            <div className="relative h-64 w-full">
                                <Image
                                    src={URL.createObjectURL(postImage)}
                                    alt="Post-image preview"
                                    fill
                                    className="object-contain"
                                />
                            </div>
                        ) : (
                            <FileUpload onChange={handlePostImageUpload} />
                        )}
                    </div>
                </div>
            </div>

            <div className="flex justify-center">
                <button
                    onClick={processImages}
                    disabled={!preImage || !postImage || isLoading}
                    className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isLoading ? "Processing..." : "Process Images"}
                </button>
            </div>

            {result && (
                <div className="border rounded-lg p-6 bg-white dark:bg-gray-800">
                    <h2 className="text-lg font-semibold mb-4">Analysis Results</h2>
                    <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded overflow-auto">
                        {JSON.stringify(result, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );
}
