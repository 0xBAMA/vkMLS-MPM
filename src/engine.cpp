#include "engine.h"

// #include <SDL.h>
// #include <SDL_vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_images.h>

#include "VkBootstrap.h"
#include <array>
#include <thread>
#include <chrono>
#include <fstream>

using namespace std::chrono_literals;

#define VMA_IMPLEMENTATION
#include <fastgltf/types.hpp>

#include "vk_mem_alloc.h"

#include <third_party/imgui/imgui.h>
#include <third_party/imgui/imgui_impl_sdl3.h>
#include <third_party/imgui/imgui_impl_vulkan.h>

#include <third_party/yaml-cpp/include/yaml-cpp/yaml.h>

#include <glm/gtx/transform.hpp>
#include <glm/gtc/packing.hpp>

#include <third_party/stb/stb_image_write.h>

inline std::string timeDateString () {
	auto now = std::chrono::system_clock::now();
	auto inTime_t = std::chrono::system_clock::to_time_t( now );
	std::stringstream ssA;
	ssA << std::put_time( std::localtime( &inTime_t ), "%Y-%m-%d at %H-%M-%S" );
	return ssA.str();
}

glm::mat2 Rotate2D ( float a ) {
	float c = cos( a ), s = sin( a );
	return glm::mat2( c, s, -s, c );
}

void PrometheusInstance::SetDebugName ( VkObjectType type, uint64_t handle, const char* name ) {
	// Must call extension functions through a function pointer:
	PFN_vkSetDebugUtilsObjectNameEXT pfnSetDebugUtilsObjectNameEXT = ( PFN_vkSetDebugUtilsObjectNameEXT ) vkGetInstanceProcAddr( instance, "vkSetDebugUtilsObjectNameEXT" );

	// // Set a name on the image
	// const VkDebugUtilsObjectNameInfoEXT imageNameInfo =
	// {
	// 	.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
	// 	.pNext = NULL,
	// 	.objectType = VK_OBJECT_TYPE_IMAGE,
	// 	.objectHandle = (uint64_t)image,
	// 	.pObjectName = "Brick Diffuse Texture",
	// };
	//
	// pfnSetDebugUtilsObjectNameEXT(device, &imageNameInfo);

	VkDebugUtilsObjectNameInfoEXT info{};
	info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	info.pNext = NULL;
	info.objectType = type;
	info.objectHandle = handle;
	info.pObjectName = name;

	pfnSetDebugUtilsObjectNameEXT( device, &info );
}

//============================================================================================================================
//============================================================================================================================
// Initialization
//============================================================================================================================
void PrometheusInstance::Init () {
	// initializing SDL
	// SDL_SetHint( SDL_HINT_VIDEO_HDR_ENABLED, "1" );
	SDL_Init( SDL_INIT_VIDEO );
	SDL_WindowFlags windowFlags = ( SDL_WindowFlags ) ( SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_FULLSCREEN );

	SDL_Rect viewRect;
	int numDisplays;
	SDL_DisplayID *displays = SDL_GetDisplays( &numDisplays );
	SDL_GetDisplayBounds( displays[ 0 ], &viewRect );

	// accumulator image is going to be 1:1 with the swapchain image
	// ImageBufferResolution.width = windowExtent.width = 3 * viewRect.w / 4;
	// ImageBufferResolution.height = windowExtent.height = 3 * viewRect.h / 4;
	globalData.accumulatorResolution.x = SimResolution.width = ImageBufferResolution.width = windowExtent.width = viewRect.w;
	globalData.accumulatorResolution.y = SimResolution.height = ImageBufferResolution.height = windowExtent.height = viewRect.h;
	SimResolution.depth = 1;

	window = SDL_CreateWindow(
		"Prometheus",
		windowExtent.width,
		windowExtent.height,
		windowFlags );

	initVulkan();
	initSwapchain();
	initCommandStructures();
	initSyncStructures();
	initResources();
	initDescriptors();
	initComputePasses();
	initImgui();
	initDefaultData();

	// everything went fine
	isInitialized = true;
}

static inline VkImageMemoryBarrier2 makeImageBarrier ( VkImage img, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess ) {
	return VkImageMemoryBarrier2 {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = srcStage,
		.srcAccessMask = srcAccess,
		.dstStageMask = dstStage,
		.dstAccessMask = dstAccess,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = img,
		.subresourceRange = {
			VK_IMAGE_ASPECT_COLOR_BIT, 0,
			VK_REMAINING_MIP_LEVELS,
			0,
			VK_REMAINING_ARRAY_LAYERS
		}
	};
}

static VkBufferMemoryBarrier2 makeBufferBarrier ( VkBuffer buf, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess ) {
	return VkBufferMemoryBarrier2 {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
		.srcStageMask = srcStage,
		.srcAccessMask = srcAccess,
		.dstStageMask = dstStage,
		.dstAccessMask = dstAccess,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = buf,
		.offset = 0,
		.size = VK_WHOLE_SIZE
	};
}

//============================================================================================================================
// Draw
//============================================================================================================================
void PrometheusInstance::Draw () {
	// wait until the gpu has finished rendering the last frame. Timeout of 1 second
	VK_CHECK( vkWaitForFences( device, 1, &getCurrentFrame().renderFence, true, 1000000000 ) );

	// we want to take this opportunity to now reset the deletion queue, since this fence marks the completion
	getCurrentFrame().deletionQueue.flush(); // of all operations which could be using the data...
	getCurrentFrame().frameDescriptors.clear_pools( device ); // mark the allocated descriptors as available

	// and now reset that fence so we can use it again, to signal this frame's completion
	VK_CHECK( vkResetFences( device, 1, &getCurrentFrame().renderFence ) );

	//request image from the swapchain
	uint32_t swapchainImageIndex;
	VkResult e = vkAcquireNextImageKHR( device, swapchain, 1000000000, getCurrentFrame().swapchainSemaphore, VK_NULL_HANDLE, &swapchainImageIndex );
	if ( e == VK_ERROR_OUT_OF_DATE_KHR ) {
		resizeRequest = true;
		return; // we will skip trying to draw the rest of the frame, because we have detected a swapchain mismatch
	}

	// Vulkan handles are aliased 64-bit pointers, basically shortens later code
	VkCommandBuffer cmd = getCurrentFrame().mainCommandBuffer;

	// because we've hit the fence, we are safe to reset the image buffer
	VK_CHECK( vkResetCommandBuffer( cmd, 0 ) );

	// begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );

	// this is for render scaling
	drawExtent.height = uint32_t( std::min( swapchainExtent.height, drawImage.imageExtent.height ) * renderScale );
	drawExtent.width = uint32_t( std::min( swapchainExtent.width, drawImage.imageExtent.width ) * renderScale );

	// update the UBO contents
	static float mouseX, mouseY;
	auto ret = SDL_GetMouseState( &mouseX, &mouseY );
	globalData.mouseLoc = glm::vec3( mouseX, mouseY, ( ret & SDL_BUTTON_LEFT ) ? 1.0f : 0.0f );
	globalData.presentBufferResolution = glm::uvec2( drawExtent.width, drawExtent.height );
	globalData.frameNumber = frameNumber;
	globalData.resolutionScalar = renderScale;
	globalData.numPoints = numPoints;
	globalData.numPointsFluid = numPointsFluid;

	// write directly from the memory on the PrometheusInstance
	GlobalData* uniformData = ( GlobalData * ) GlobalUBO.allocation->GetMappedData();
	*uniformData = globalData;

	// start the command buffer recording
	VK_CHECK( vkBeginCommandBuffer( cmd, &cmdBeginInfo ) );

	// put the core images into a general format
	vkutil::transition_image( cmd, Accumulator.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );
	vkutil::transition_image( cmd, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );
	vkutil::transition_image( cmd, velocityXAtomic.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );
	vkutil::transition_image( cmd, velocityYAtomic.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );
	vkutil::transition_image( cmd, massAtomic.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );
	vkutil::transition_image( cmd, resolvedAtomics.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL );

	if ( globalData.reset != 0 ) {
		globalData.reset = 0;

		// Precomputing the volume term for the points
		initPoints();
		VkBufferMemoryBarrier2 barrier = makeBufferBarrier( pointBuffer.buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT );
		VkDependencyInfo barrierDependency {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.bufferMemoryBarrierCount = 1,
			.pBufferMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2( cmd, &barrierDependency );

		BufferClears( cmd );
		PointToGrid.invoke( cmd );
		EstimateVolume.invoke( cmd );
	} else {
		for ( int i = 0; i < iterations; ++i ) {
			// main sim loop - 4 steps + the additional 2 fluid steps
			BufferClears( cmd );
			PointToGrid.invoke( cmd );
			PointToGridFluidPass1.invoke( cmd );
			PointToGridFluidPass2.invoke( cmd );
			UpdateGrid.invoke( cmd );
			GridToPoint.invoke( cmd );
		}
	}

	// Point drawing
	PointRaster.invoke( cmd );

	// compute shader to accumulate the raster result + put the resolved final image into the drawImage...
	BufferPresent.invoke( cmd );

	// transition the images for the copy
	vkutil::transition_image( cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL );
	vkutil::transition_image( cmd, swapchainImages[ swapchainImageIndex ], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );

	// execute a copy from the draw image into the swapchain
	vkutil::copy_image_to_image( cmd, drawImage.image, swapchainImages[ swapchainImageIndex ], drawExtent, swapchainExtent );

	// set swapchain image layout to Attachment Optimal so we can draw it
	vkutil::transition_image( cmd, swapchainImages[ swapchainImageIndex ], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL );

	//draw imgui into the swapchain image
	drawImgui( cmd, swapchainImageViews[ swapchainImageIndex ] );

	// transition the image from layout general to ready-for-swapchain-handoff
	vkutil::transition_image( cmd, swapchainImages[ swapchainImageIndex ], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR );

	// Kill recording, and put it in "executable" state
	VK_CHECK( vkEndCommandBuffer( cmd ) );

	// before submitting to the queue, we need to specify the specific dependencies
	// we want to wait on the presentSemaphore, signaled when the swapchain is ready
	// we will signal the renderSemaphore, when rendering has finished
	VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info( cmd );
	VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info( VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, getCurrentFrame().swapchainSemaphore );
	VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info( VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, swapchainPresentSemaphores[ swapchainImageIndex ] );

	VkSubmitInfo2 submit = vkinit::submit_info( &cmdinfo, &signalInfo, &waitInfo );

	// submit command buffer to the queue and execute it... renderFence will now block until it finishes
	VK_CHECK( vkQueueSubmit2( graphicsQueue, 1, &submit, getCurrentFrame().renderFence ) );

	// swapchain present to visible window...
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;
	presentInfo.pSwapchains = &swapchain;
	presentInfo.swapchainCount = 1;
	// wait on renderSemaphore, to tell when we are finished preparing the image
	presentInfo.pWaitSemaphores = &swapchainPresentSemaphores[ swapchainImageIndex ];
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pImageIndices = &swapchainImageIndex;

	VkResult presentResult = vkQueuePresentKHR( graphicsQueue, &presentInfo );
	if ( presentResult == VK_ERROR_OUT_OF_DATE_KHR ) {
		resizeRequest = true; // swapchain mismatch
	}

	//increase the number of frames drawn
	frameNumber++;
}

//============================================================================================================================
// Main Loop
//============================================================================================================================
void PrometheusInstance::MainLoop () {
	SDL_Event e;

	bool quit = false;
	while ( !quit ) {
		// event handling loop
		while ( SDL_PollEvent( &e ) ) {
			ImGui_ImplSDL3_ProcessEvent( &e );

			const bool shift = SDL_GetModState() & SDL_KMOD_LSHIFT;
			const float amount = shift ? 0.1f : 0.01f;

			if ( e.type == SDL_EVENT_QUIT ) {
				quit = true;
			}

			if ( e.type == SDL_EVENT_KEY_UP && e.key.scancode == SDL_SCANCODE_ESCAPE && shift ) {
				quit = true;
			}

			const bool* kb = SDL_GetKeyboardState( NULL );
			if ( kb[ SDL_SCANCODE_R ] ) {
				globalData.reset = 1;
			}

			if ( kb[ SDL_SCANCODE_T ] && shift ) {
				screenshot();
			}
		}

		// handling minimized application
		if ( stopRendering ) {
			// throttle the speed to avoid busy loop
			std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
		} else {
			// imgui new frame
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplSDL3_NewFrame();
			ImGui::NewFrame();

			// some imgui UI to test
			// ImGui::ShowDemoWindow();

			if ( ImGui::Begin( "Controls" ) ) {
				ImGui::SliderFloat( "Elastic Mu", &globalData.elasticMu, 0.001f, 40.0f );
				ImGui::SliderFloat( "Elastic Lambda", &globalData.elasticLambda, 0.001f, 40.0f );
				ImGui::Separator();
				ImGui::SliderFloat( "Fluid Rest Density", &globalData.restDensity, 0.001f, 10.0f );
				ImGui::SliderFloat( "Fluid Dynamic Viscosity", &globalData.dynamicViscosity, 0.001f, 1.0f );
				ImGui::SliderFloat( "Fluid Equation of State Stiffness", &globalData.eosStiffness, 0.001f, 10.0f );
				ImGui::SliderFloat( "Fluid Equation of State Power", &globalData.eosPower, 0.001f, 10.0f );
				ImGui::Separator();
				ImGui::SliderFloat( "Mouse Size", &globalData.mouseSize, 0.001f, 100.0f );
				ImGui::SliderFloat( "Mouse Force", &globalData.mouseForceScalar, 0.01f, 2.0f );
				ImGui::SliderFloat( "Gravity", &globalData.gravityScalar, -0.01f, 0.01f, "%.5f" );
				ImGui::SliderFloat( "Timestep", &globalData.dt, 0.001f, 0.1f );
				ImGui::SliderInt( "Iterations Per Frame", &iterations, 1, 25 );

				ImGui::SeparatorText( "Inset View" );
				ImGui::SliderFloat( "Scale", &scalar, 1.0f, 10.0f );
				ImGui::SliderInt( "Width", &windowWidth, 10.0f, 1600.0f );
				ImGui::SliderInt( "Height", &windowHeight, 10.0f, 1600.0f );
				ImGui::SliderInt( "X Offset", &xOffset, -10000.0f, 10000.0f );
				ImGui::SliderInt( "Y Offset", &yOffset, -10000.0f, 10000.0f );
			}
			ImGui::End();

			// make imgui calculate internal draw structures
			ImGui::Render();

			// we're ready to draw the next frame
			Draw();
		}
	}

	// checking to see if we have flagged a window resize
	if ( resizeRequest ) {
		resizeSwapchain();
	}
}

//============================================================================================================================
// Cleanup
//============================================================================================================================
void PrometheusInstance::ShutDown () {
	// if we successfully made it through init
	if ( isInitialized ) {
		// make sure the gpu has stopped all work
		vkDeviceWaitIdle( device );

		// kill frameData
		for ( int i = 0; i < FRAME_OVERLAP; i++ ) {
			// killing the command pool implicitly kills the command buffers
			vkDestroyCommandPool( device, frameData[ i ].commandPool, nullptr );

			// destroy sync objects
			vkDestroyFence( device, frameData[ i ].renderFence, nullptr );
			vkDestroySemaphore( device, frameData[ i ].swapchainSemaphore, nullptr );

			// delete any remaining per-frame resources...
			frameData[ i ].deletionQueue.flush();
		}

		for ( auto& s : swapchainPresentSemaphores ) {
			vkDestroySemaphore( device, s, nullptr );
		}

		// destroy any remaining global resources
		mainDeletionQueue.flush();

		// destroy remaining resources
		destroySwapchain();
		vkDestroySurfaceKHR( instance, surface, nullptr );
		vkDestroyDevice( device, nullptr );
		vkb::destroy_debug_utils_messenger( instance, debugMessenger );
		vkDestroyInstance( instance, nullptr );
		SDL_DestroyWindow( window );
	}
}

//===========================================================================================================================
// Helpers
//===========================================================================================================================
void PrometheusInstance::initVulkan () {
	// make the vulkan instance, with basic debug features
	vkb::InstanceBuilder builder;
	auto inst_ret = builder.set_app_name( "Prometheus" )
		.request_validation_layers( useValidationLayers )
		.use_default_debug_messenger()
		.require_api_version( 1, 3, 0 )
		.build();

	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance
	instance = vkb_inst.instance;
	debugMessenger = vkb_inst.debug_messenger;

	// create a surface to render to
	SDL_Vulkan_CreateSurface( window, instance, NULL, &surface );

	//vulkan 1.3 features
	VkPhysicalDeviceVulkan13Features features13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
	features13.dynamicRendering = true;
	features13.synchronization2 = true;

	//vulkan 1.2 features
	VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	features12.bufferDeviceAddress = true;
	features12.descriptorIndexing = true;

	//use vkbootstrap to select a gpu.
	//We want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDeviceSelect = selector
		.set_minimum_version( 1, 3 )
		.set_required_features_13( features13 )
		.set_required_features_12( features12 )
		.set_surface( surface )
		.select()
		.value();

	//create the final vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDeviceSelect };
	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a vulkan application
	device = vkbDevice.device;
	physicalDevice = physicalDeviceSelect.physical_device;

	{
		// reporting some platform info
		VkPhysicalDeviceProperties temp;
		vkGetPhysicalDeviceProperties( vkbDevice.physical_device, &temp );

		std::string GPUType;
		switch ( temp.deviceType ) {
			case VK_PHYSICAL_DEVICE_TYPE_OTHER: GPUType = "Other GPU"; break;
			case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: GPUType = "Integrated GPU"; break;
			case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: GPUType = "Discrete GPU"; break;
			case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: GPUType = "Virtual GPU"; break;
			case VK_PHYSICAL_DEVICE_TYPE_CPU: GPUType = "CPU as GPU"; break;
			default: GPUType = "Unknown"; break;
		}
		fmt::print( "Running on {} ({})", temp.deviceName, GPUType );
		fmt::print( "\n\nDevice Limits:\n" );
		// fmt::print( "{}\n" );
		// fmt::print( "{}\n" );
		fmt::print( "Max Push Constant Size: {}\n", temp.limits.maxPushConstantsSize );
		fmt::print( "Max Compute Workgroup Size: {}x {}y {}z\n", temp.limits.maxComputeWorkGroupSize[ 0 ], temp.limits.maxComputeWorkGroupSize[ 1 ], temp.limits.maxComputeWorkGroupSize[ 2 ] );
		fmt::print( "Max Compute Workgroup Invocations (single workgroup): {}\n", temp.limits.maxComputeWorkGroupInvocations );
		fmt::print( "Max Compute Workgroup Count: {}x {}y {}z\n", temp.limits.maxComputeWorkGroupCount[ 0 ], temp.limits.maxComputeWorkGroupCount[ 1 ], temp.limits.maxComputeWorkGroupCount[ 2 ] );
		fmt::print( "Max Compute Shared Memory Size: {}\n\n", temp.limits.maxComputeSharedMemorySize );
		fmt::print( "Max Storage Buffer Range: {}\n", temp.limits.maxStorageBufferRange );
		fmt::print( "Max Framebuffer Width: {}\n", temp.limits.maxFramebufferWidth );
		fmt::print( "Max Framebuffer Height: {}\n", temp.limits.maxFramebufferHeight );
		fmt::print( "Max Image Dimension(1D): {}\n", temp.limits.maxImageDimension1D );
		fmt::print( "Max Image Dimension(2D): {}\n", temp.limits.maxImageDimension2D );
		fmt::print( "Max Image Dimension(3D): {}\n", temp.limits.maxImageDimension3D );
		fmt::print( "\n\n" );
	}

	// use vkbootstrap to get a Graphics queue
	graphicsQueue = vkbDevice.get_queue( vkb::QueueType::graphics ).value();
	graphicsQueueFamilyIndex = vkbDevice.get_queue_index( vkb::QueueType::graphics ).value();

	// initialize the memory allocator
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = physicalDevice;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator( &allocatorInfo, &allocator );

	mainDeletionQueue.push_function( [ & ] () {
		vmaDestroyAllocator( allocator ); // first example of deletion queue...
	});
}

void PrometheusInstance::initSwapchain () {
	createSwapchain( windowExtent.width, windowExtent.height );
}

void PrometheusInstance::initCommandStructures () {
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info( graphicsQueueFamilyIndex, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	for ( int i = 0; i < FRAME_OVERLAP; i++ ) {
		// create a command pool allocator
		VK_CHECK( vkCreateCommandPool( device, &commandPoolInfo, nullptr, &frameData[ i ].commandPool ) );

		// and a command buffer from that command pool
		VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info( frameData[ i ].commandPool, 1 );
		VK_CHECK( vkAllocateCommandBuffers( device, &cmdAllocInfo, &frameData[ i ].mainCommandBuffer ) );
	}
	VK_CHECK( vkCreateCommandPool( device, &commandPoolInfo, nullptr, &immediateCommandPool ) );

	// allocating the command buffer for immediate submits
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info( immediateCommandPool, 1 );
	VK_CHECK( vkAllocateCommandBuffers( device, &cmdAllocInfo, &immediateCommandBuffer ) );

	mainDeletionQueue.push_function( [ = ] ()  {
		vkDestroyCommandPool( device, immediateCommandPool, nullptr );
	});
}

void PrometheusInstance::initSyncStructures () {
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info( VK_FENCE_CREATE_SIGNALED_BIT );
	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();
	for ( int i = 0; i < FRAME_OVERLAP; i++ ) {
	// we need to create one fence ( frame end mark )
		VK_CHECK( vkCreateFence( device, &fenceCreateInfo, nullptr, &frameData[ i ].renderFence ) );
	// and two semaphores: swapchain image ready, and render finished
		VK_CHECK( vkCreateSemaphore( device, &semaphoreCreateInfo, nullptr, &frameData[ i ].swapchainSemaphore ) );
	}

	swapchainPresentSemaphores.resize( swapchainImages.size() );
	for ( size_t i = 0; i < swapchainImages.size(); i++ ) {
		VK_CHECK( vkCreateSemaphore( device, &semaphoreCreateInfo, nullptr, &swapchainPresentSemaphores[ i ] ) );
	}

	VK_CHECK( vkCreateFence( device, &fenceCreateInfo, nullptr, &immediateFence ) );
	mainDeletionQueue.push_function( [ = ] ()  { vkDestroyFence( device, immediateFence, nullptr ); } );

	// will also need several barriers for the compute/graphics operations
}

void PrometheusInstance::initDescriptors  () {
	//create a descriptor pool that will hold 10 sets with some different contents
	std::vector< DescriptorAllocatorGrowable::PoolSizeRatio > sizes = {
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6 },
	};

	globalDescriptorAllocator.init( device, 10, sizes );

	//make sure both the descriptor allocator and the new layout get cleaned up properly
	mainDeletionQueue.push_function( [ & ] () {
		globalDescriptorAllocator.destroy_pools( device );
	});

	for ( int i = 0; i < FRAME_OVERLAP; i++ ) {
		// create a descriptor pool
		std::vector< DescriptorAllocatorGrowable::PoolSizeRatio > frameSizes = {
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
		};

		frameData[ i ].frameDescriptors = DescriptorAllocatorGrowable{};
		frameData[ i ].frameDescriptors.init( device, 1000, frameSizes );

		mainDeletionQueue.push_function([ &, i ]() {
			frameData[ i ].frameDescriptors.destroy_pools( device );
		});
	}
}

void PrometheusInstance::initResources () {

	// API resource allocation:
	// create the buffer for the UBO
	{
		GlobalUBO = createBuffer( sizeof( GlobalData ), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU );
		SetDebugName( VK_OBJECT_TYPE_BUFFER, ( uint64_t ) GlobalUBO.buffer, "Global Data UBO" );
	}

	// create the accumulator texture
	{
		VkExtent3D bufferExtent = { globalData.accumulatorResolution.x, globalData.accumulatorResolution.y, 1 };
		Accumulator = createImage( bufferExtent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT );
		SetDebugName( VK_OBJECT_TYPE_IMAGE, ( uint64_t ) Accumulator.image, "Accumulator" );
	}

	{
		// buffers used in particle-to-grid
		velocityXAtomic	= createImage( SimResolution, VK_FORMAT_R32_SINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT );
		SetDebugName( VK_OBJECT_TYPE_IMAGE, ( uint64_t ) velocityXAtomic.image, "X Velocity Accumulator" );
		velocityYAtomic	= createImage( SimResolution, VK_FORMAT_R32_SINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT );
		SetDebugName( VK_OBJECT_TYPE_IMAGE, ( uint64_t ) velocityYAtomic.image, "Y Velocity Accumulator" );
		massAtomic		= createImage( SimResolution, VK_FORMAT_R32_SINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT );
		SetDebugName( VK_OBJECT_TYPE_IMAGE, ( uint64_t ) massAtomic.image, "Mass Accumulator" );
		resolvedAtomics	= createImage( SimResolution, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT );
		SetDebugName( VK_OBJECT_TYPE_IMAGE, ( uint64_t ) resolvedAtomics.image, "Atomic Resolve Texture" );
	}

	// make sure to clean up at the end
	mainDeletionQueue.push_function([ & ] () {
		// destroying buffers
		destroyBuffer( GlobalUBO );

		// destroying images
		destroyImage( Accumulator );
		destroyImage( velocityXAtomic );
		destroyImage( velocityYAtomic );
		destroyImage( massAtomic );
		destroyImage( resolvedAtomics );
	});
}

void PrometheusInstance::initComputePasses () {

// Simulation
	{ // Estimate Volume - runs per point
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the point buffer
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // three textures used for sim
			builder.add_binding( 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			EstimateVolume.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) EstimateVolume.descriptorSetLayout, "Volume Estimate Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &EstimateVolume.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &EstimateVolume.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) EstimateVolume.pipelineLayout, "Volume Estimate Pipeline Layout" );

			VkShaderModule EstimateVolumeShader;
			if ( !vkutil::load_shader_module("../shaders/estimateVolume.comp.glsl.spv", device, &EstimateVolumeShader ) ) {
				fmt::print( "Error when building the Volume Estimate Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) EstimateVolumeShader, "Volume Estimate Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = EstimateVolumeShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = EstimateVolume.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &EstimateVolume.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) EstimateVolume.pipeline, "Volume Estimate Compute Pipeline" );
			vkDestroyShaderModule( device, EstimateVolumeShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, EstimateVolume.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, EstimateVolume.pipelineLayout, nullptr );
				vkDestroyPipeline( device, EstimateVolume.pipeline, nullptr );
			});
		}

		// invoke() lambda
		EstimateVolume.invoke = [ & ] ( VkCommandBuffer cmd ){
			// dynamic descriptor allocation, to bind a texture
			EstimateVolume.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, EstimateVolume.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.write_image( 2, velocityXAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 3, velocityYAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 4, massAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.update_set( device, EstimateVolume.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, EstimateVolume.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, EstimateVolume.pipelineLayout, 0, 1, &EstimateVolume.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			EstimateVolume.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, EstimateVolume.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &EstimateVolume.pushConstants );

			// dispatch for all the pixels
			vkCmdDispatch( cmd, ( numPoints ) / 64, 1, 1 );

			VkBufferMemoryBarrier2 barrier = makeBufferBarrier( pointBuffer.buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT );
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.bufferMemoryBarrierCount = 1,
				.pBufferMemoryBarriers = &barrier,
			};
			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}

	{ // PointToGrid - runs per point
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the point buffer
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // three textures used for sim
			builder.add_binding( 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			PointToGrid.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) PointToGrid.descriptorSetLayout, "Point To Grid Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &PointToGrid.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &PointToGrid.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) PointToGrid.pipelineLayout, "Point To Grid Pipeline Layout" );

			VkShaderModule PointToGridShader;
			if ( !vkutil::load_shader_module("../shaders/pointToGrid.comp.glsl.spv", device, &PointToGridShader ) ) {
				fmt::print( "Error when building the Point To Grid Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) PointToGridShader, "Point To Grid Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = PointToGridShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = PointToGrid.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &PointToGrid.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) PointToGrid.pipeline, "Point To Grid Compute Pipeline" );
			vkDestroyShaderModule( device, PointToGridShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, PointToGrid.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, PointToGrid.pipelineLayout, nullptr );
				vkDestroyPipeline( device, PointToGrid.pipeline, nullptr );
			});
		}

		// invoke() lambda
		PointToGrid.invoke = [ & ] ( VkCommandBuffer cmd ){
			// dynamic descriptor allocation, to bind a texture
			PointToGrid.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, PointToGrid.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.write_image( 2, velocityXAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 3, velocityYAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 4, massAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.update_set( device, PointToGrid.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PointToGrid.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PointToGrid.pipelineLayout, 0, 1, &PointToGrid.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			PointToGrid.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, PointToGrid.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &PointToGrid.pushConstants );

			// dispatch for all the pixels
			vkCmdDispatch( cmd, ( numPoints ) / 64, 1, 1 );

			// writes to the buffers must complete
			VkImageMemoryBarrier2 barriers[ 3 ] = {
				makeImageBarrier( velocityXAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( velocityYAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( massAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT )
			};
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 3,
				.pImageMemoryBarriers = barriers
			};
			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}

	{ // PointToGridFluidPass1 - runs per point
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the point buffer
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // three textures used for sim
			builder.add_binding( 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			PointToGridFluidPass1.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) PointToGridFluidPass1.descriptorSetLayout, "Point To Grid Fluid Pass 1 Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &PointToGridFluidPass1.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &PointToGridFluidPass1.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) PointToGridFluidPass1.pipelineLayout, "Point To Grid Fluid Pass 1 Pipeline Layout" );

			VkShaderModule PointToGridShader;
			if ( !vkutil::load_shader_module("../shaders/pointToGridFluidPass1.comp.glsl.spv", device, &PointToGridShader ) ) {
				fmt::print( "Error when building the Point To Grid Fluid Pass 1 Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) PointToGridShader, "Point To Grid Fluid Pass 1 Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = PointToGridShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = PointToGridFluidPass1.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &PointToGridFluidPass1.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) PointToGridFluidPass1.pipeline, "Point To Grid Fluid Pass 1 Compute Pipeline" );
			vkDestroyShaderModule( device, PointToGridShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, PointToGridFluidPass1.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, PointToGridFluidPass1.pipelineLayout, nullptr );
				vkDestroyPipeline( device, PointToGridFluidPass1.pipeline, nullptr );
			});
		}

		// invoke() lambda
		PointToGridFluidPass1.invoke = [ & ] ( VkCommandBuffer cmd ){
			// dynamic descriptor allocation, to bind a texture
			PointToGridFluidPass1.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, PointToGridFluidPass1.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.write_image( 2, velocityXAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 3, velocityYAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 4, massAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.update_set( device, PointToGridFluidPass1.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PointToGridFluidPass1.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PointToGridFluidPass1.pipelineLayout, 0, 1, &PointToGridFluidPass1.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			PointToGridFluidPass1.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, PointToGridFluidPass1.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &PointToGridFluidPass1.pushConstants );

			// dispatch for all the pixels
			vkCmdDispatch( cmd, ( numPointsFluid ) / 64, 1, 1 );

			// writes to the buffers must complete
			VkImageMemoryBarrier2 barriers[ 3 ] = {
				makeImageBarrier( velocityXAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( velocityYAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( massAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT )
			};
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 3,
				.pImageMemoryBarriers = barriers
			};
			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}

	{ // PointToGridFluidPass2 - runs per point
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the point buffer
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // three textures used for sim
			builder.add_binding( 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			PointToGridFluidPass2.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) PointToGridFluidPass2.descriptorSetLayout, "Point To Grid Fluid Pass 2 Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &PointToGridFluidPass2.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &PointToGridFluidPass2.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) PointToGridFluidPass2.pipelineLayout, "Point To Grid Fluid Pass 2 Pipeline Layout" );

			VkShaderModule PointToGridShader;
			if ( !vkutil::load_shader_module("../shaders/pointToGridFluidPass2.comp.glsl.spv", device, &PointToGridShader ) ) {
				fmt::print( "Error when building the Point To Grid Fluid Pass 2 Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) PointToGridShader, "Point To Grid Fluid Pass 2 Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = PointToGridShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = PointToGridFluidPass2.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &PointToGridFluidPass2.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) PointToGridFluidPass2.pipeline, "Point To Grid Fluid Pass 2 Compute Pipeline" );
			vkDestroyShaderModule( device, PointToGridShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, PointToGridFluidPass2.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, PointToGridFluidPass2.pipelineLayout, nullptr );
				vkDestroyPipeline( device, PointToGridFluidPass2.pipeline, nullptr );
			});
		}

		// invoke() lambda
		PointToGridFluidPass2.invoke = [ & ] ( VkCommandBuffer cmd ){
			// dynamic descriptor allocation, to bind a texture
			PointToGridFluidPass2.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, PointToGridFluidPass2.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.write_image( 2, velocityXAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 3, velocityYAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 4, massAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.update_set( device, PointToGridFluidPass2.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PointToGridFluidPass2.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PointToGridFluidPass2.pipelineLayout, 0, 1, &PointToGridFluidPass2.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			PointToGridFluidPass2.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, PointToGridFluidPass2.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &PointToGridFluidPass2.pushConstants );

			// dispatch for all the pixels
			vkCmdDispatch( cmd, ( numPointsFluid ) / 64, 1, 1 );

			// writes to the buffers must complete
			VkImageMemoryBarrier2 barriers[ 3 ] = {
				makeImageBarrier( velocityXAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( velocityYAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( massAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT )
			};
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 3,
				.pImageMemoryBarriers = barriers
			};
			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}

	{ // UpdateGrid - runs per grid cell
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the point buffer
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // three textures used for sim
			builder.add_binding( 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			UpdateGrid.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) UpdateGrid.descriptorSetLayout, "Update Grid Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &UpdateGrid.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &UpdateGrid.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) UpdateGrid.pipelineLayout, "Update Grid Pipeline Layout" );

			VkShaderModule UpdateGridShader;
			if ( !vkutil::load_shader_module("../shaders/updateGrid.comp.glsl.spv", device, &UpdateGridShader ) ) {
				fmt::print( "Error when building the Update Grid Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) UpdateGridShader, "Update Grid Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = UpdateGridShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = UpdateGrid.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &UpdateGrid.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) UpdateGrid.pipeline, "Update Grid Compute Pipeline" );
			vkDestroyShaderModule( device, UpdateGridShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, UpdateGrid.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, UpdateGrid.pipelineLayout, nullptr );
				vkDestroyPipeline( device, UpdateGrid.pipeline, nullptr );
			});
		}

		// invoke() lambda
		UpdateGrid.invoke = [ & ] ( VkCommandBuffer cmd ){
			// dynamic descriptor allocation, to bind a texture
			UpdateGrid.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, UpdateGrid.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.write_image( 2, velocityXAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 3, velocityYAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 4, massAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 5, resolvedAtomics.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.update_set( device, UpdateGrid.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, UpdateGrid.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, UpdateGrid.pipelineLayout, 0, 1, &UpdateGrid.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			UpdateGrid.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, UpdateGrid.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &UpdateGrid.pushConstants );

			// dispatch for all the pixels
			vkCmdDispatch( cmd, ( SimResolution.width + 15 ) / 16, ( SimResolution.height + 15 ) / 16, 1 );

			// writes to the buffers must complete
			VkImageMemoryBarrier2 barriers[ 3 ] = {
				makeImageBarrier( velocityXAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( velocityYAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
				makeImageBarrier( massAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT )
			};
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 3,
				.pImageMemoryBarriers = barriers
			};
			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}

	{ // GridToPoint - runs per point
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the point buffer
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // three textures used for sim
			builder.add_binding( 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			builder.add_binding( 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
			GridToPoint.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) GridToPoint.descriptorSetLayout, "Grid To Point Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &GridToPoint.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &GridToPoint.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) GridToPoint.pipelineLayout, "Grid To Point Pipeline Layout" );

			VkShaderModule GridToPointShader;
			if ( !vkutil::load_shader_module("../shaders/gridToPoint.comp.glsl.spv", device, &GridToPointShader ) ) {
				fmt::print( "Error when building the Grid To Point Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) GridToPointShader, "Grid To Point Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = GridToPointShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = GridToPoint.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &GridToPoint.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) GridToPoint.pipeline, "Grid To Point Compute Pipeline" );
			vkDestroyShaderModule( device, GridToPointShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, GridToPoint.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, GridToPoint.pipelineLayout, nullptr );
				vkDestroyPipeline( device, GridToPoint.pipeline, nullptr );
			});
		}

		// invoke() lambda
		GridToPoint.invoke = [ & ] ( VkCommandBuffer cmd ){
			// dynamic descriptor allocation, to bind a texture
			GridToPoint.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, GridToPoint.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.write_image( 2, velocityXAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 3, velocityYAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 4, massAtomic.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 5, resolvedAtomics.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.update_set( device, GridToPoint.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, GridToPoint.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, GridToPoint.pipelineLayout, 0, 1, &GridToPoint.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			GridToPoint.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, GridToPoint.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &GridToPoint.pushConstants );

			// dispatch for all the pixels
			vkCmdDispatch( cmd, ( numPoints + numPointsFluid ) / 64, 1, 1 );

			VkBufferMemoryBarrier2 barrier = makeBufferBarrier( pointBuffer.buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT );
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.bufferMemoryBarrierCount = 1,
				.pBufferMemoryBarriers = &barrier,
			};
			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}

	{ // point raster
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ); // the SSBO with point locations
			PointRaster.descriptorSetLayout = builder.build( device,  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT );
		}

		{ // pipeline layout + pipeline build
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo rasterLayout{};
			rasterLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			rasterLayout.pNext = nullptr;
			rasterLayout.pSetLayouts = &PointRaster.descriptorSetLayout;
			rasterLayout.setLayoutCount = 1;
			rasterLayout.pPushConstantRanges = &pushConstant;
			rasterLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &rasterLayout, nullptr, &PointRaster.pipelineLayout ) );

			VkShaderModule pointSpriteFragShader;
			if ( !vkutil::load_shader_module( "../shaders/pointRaster.frag.glsl.spv", device, &pointSpriteFragShader ) ) {
				fmt::print( "Error when building the Point Raster fragment shader module\n" );
			} else {
				fmt::print( "Point Raster fragment shader successfully loaded\n" );
			}

			VkShaderModule pointSpriteVertexShader;
			if ( !vkutil::load_shader_module( "../shaders/pointRaster.vert.glsl.spv", device, &pointSpriteVertexShader ) ) {
				fmt::print( "Error when building the Point Raster vertex shader module\n" );
			} else {
				fmt::print( "Point Raster vertex shader successfully loaded\n" );
			}

			PipelineBuilder pipelineBuilder;
			pipelineBuilder._pipelineLayout = PointRaster.pipelineLayout;
			pipelineBuilder.set_shaders( pointSpriteVertexShader, pointSpriteFragShader );
			pipelineBuilder.set_input_topology( VK_PRIMITIVE_TOPOLOGY_POINT_LIST );
			pipelineBuilder.set_polygon_mode( VK_POLYGON_MODE_FILL );
			pipelineBuilder.set_cull_mode( VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE );
			pipelineBuilder.set_multisampling_none();
			pipelineBuilder.disable_blending();
			pipelineBuilder.disable_depthtest();
			pipelineBuilder.set_color_attachment_format( Accumulator.imageFormat );
			PointRaster.pipeline = pipelineBuilder.build_pipeline( device );

			// cleanup
			vkDestroyShaderModule( device, pointSpriteFragShader, nullptr );
			vkDestroyShaderModule( device, pointSpriteVertexShader, nullptr );

			mainDeletionQueue.push_function( [ & ] ()  {
				vkDestroyDescriptorSetLayout( device, PointRaster.descriptorSetLayout, nullptr );
				vkDestroyPipeline( device, PointRaster.pipeline, nullptr );
				vkDestroyPipelineLayout( device, PointRaster.pipelineLayout, nullptr );
			});
		}

		PointRaster.invoke = [ & ] ( VkCommandBuffer cmd ) {
			// additive raster for the agent locations
			VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
			VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info( Accumulator.imageView, &clearColor, VK_IMAGE_LAYOUT_GENERAL );
			VkRenderingInfo renderInfo = vkinit::rendering_info( ImageBufferResolution, &colorAttachment, nullptr );

			vkCmdBeginRendering( cmd, &renderInfo );
			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PointRaster.pipeline );

			// dynamic descriptor allocation, to bind a texture
			PointRaster.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, PointRaster.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_buffer( 1, pointBuffer.buffer, ( numPoints + numPointsFluid ) * sizeof( point ), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER );
				writer.update_set( device, PointRaster.descriptorSet );
			}

			//set dynamic viewport and scissor
			VkViewport viewport = {};
			viewport.x = 0;
			viewport.y = 0;
			viewport.width = float( ImageBufferResolution.width );
			viewport.height = float( ImageBufferResolution.height );
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport( cmd, 0, 1, &viewport );

			VkRect2D scissor = {};
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			scissor.extent.width = ImageBufferResolution.width;
			scissor.extent.height = ImageBufferResolution.height;
			vkCmdSetScissor( cmd, 0, 1, &scissor );

			PointRaster.pushConstants.pointScale = 1.0f;
			PointRaster.pushConstants.wangSeed = genWangSeed();

			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,  PointRaster.pipelineLayout, 0, 1, &PointRaster.descriptorSet, 0, nullptr );
			vkCmdPushConstants( cmd, PointRaster.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &PointRaster.pushConstants );

			vkCmdDraw( cmd, numPoints + numPointsFluid, 1, 0, 0 );

			// //set dynamic viewport and scissor
			viewport.x = scalar * ( -globalData.mouseLoc.x - windowWidth / 2.0f ) + xOffset;
			viewport.y = scalar * ( -globalData.mouseLoc.y - windowHeight / 2.0f ) + yOffset;
			viewport.width = scalar * ImageBufferResolution.width;
			viewport.height = scalar * ImageBufferResolution.height;
			vkCmdSetViewport( cmd, 0, 1, &viewport );

			scissor.offset.x = ImageBufferResolution.width - windowWidth;
			scissor.offset.y = ImageBufferResolution.height - windowHeight;
			scissor.extent.width = windowWidth;
			scissor.extent.height = windowHeight;
			vkCmdSetScissor( cmd, 0, 1, &scissor );

			/*
			VkClearAttachment clearAttachment = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.colorAttachment = 0,
				.clearValue = {{0.0f, 0.0f, 0.0f, 0.2f}}
			};

			VkClearRect clearRect = {
				.rect = {
					.offset = scissor.offset,
					.extent = scissor.extent
				},
				.baseArrayLayer = 0,
				.layerCount = 1
			};

			vkCmdClearAttachments(
				cmd,
				1, &clearAttachment,
				1, &clearRect
			);
			*/

			PointRaster.pushConstants.pointScale = 5.0f;
			PointRaster.pushConstants.wangSeed = genWangSeed();

			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,  PointRaster.pipelineLayout, 0, 1, &PointRaster.descriptorSet, 0, nullptr );
			vkCmdPushConstants( cmd, PointRaster.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &PointRaster.pushConstants );

			// launch a draw command to do the fullscreen triangle
			vkCmdDraw( cmd, numPoints + numPointsFluid, 1, 0, 0 );

			vkCmdEndRendering( cmd );

			VkImageMemoryBarrier2 barrierC {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

				.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
				.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

				.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,

				// now the blur has finished, we are using the filtered reads until the next agent raster
				.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,

				.image = Accumulator.image,
				.subresourceRange = {
					VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
				}
			};

			VkImageMemoryBarrier2 barriers[] = { barrierC };
			VkDependencyInfo barrierDependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = barriers
			};

			vkCmdPipelineBarrier2( cmd, &barrierDependency );
		};
	}


	{ // Present
		{ // descriptor layout
			DescriptorLayoutBuilder builder;
			builder.add_binding( 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ); // global config UBO
			builder.add_binding( 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ); // draw image
			builder.add_binding( 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ); // XYZ Buffer -> linear filter
			BufferPresent.descriptorSetLayout = builder.build( device, VK_SHADER_STAGE_COMPUTE_BIT );
			SetDebugName( VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, ( uint64_t ) BufferPresent.descriptorSetLayout, "Buffer Present Descriptor Set Layout" );
		}

		{ // pipeline layout + compute pipeline
			VkPushConstantRange pushConstant{};
			pushConstant.offset = 0;
			pushConstant.size = sizeof( PushConstants );
			pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

			VkPipelineLayoutCreateInfo computeLayout{};
			computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			computeLayout.pNext = nullptr;
			computeLayout.pSetLayouts = &BufferPresent.descriptorSetLayout;
			computeLayout.setLayoutCount = 1;
			computeLayout.pPushConstantRanges = &pushConstant;
			computeLayout.pushConstantRangeCount = 1;

			VK_CHECK( vkCreatePipelineLayout( device, &computeLayout, nullptr, &BufferPresent.pipelineLayout ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE_LAYOUT, ( uint64_t ) BufferPresent.pipelineLayout, "Buffer Present Pipeline Layout" );

			VkShaderModule BufferPresentShader;
			if ( !vkutil::load_shader_module("../shaders/bufferPresent.comp.glsl.spv", device, &BufferPresentShader ) ) {
				fmt::print( "Error when building the Buffer Present Compute Shader\n" );
			}
			SetDebugName( VK_OBJECT_TYPE_SHADER_MODULE, ( uint64_t ) BufferPresentShader, "Buffer Present Shader Module" );

			VkPipelineShaderStageCreateInfo stageinfo{};
			stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stageinfo.pNext = nullptr;
			stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			stageinfo.module = BufferPresentShader;
			stageinfo.pName = "main";

			VkComputePipelineCreateInfo computePipelineCreateInfo{};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.pNext = nullptr;
			computePipelineCreateInfo.layout = BufferPresent.pipelineLayout;
			computePipelineCreateInfo.stage = stageinfo;

			VK_CHECK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &BufferPresent.pipeline ) );
			SetDebugName( VK_OBJECT_TYPE_PIPELINE, ( uint64_t ) BufferPresent.pipeline, "Buffer Present Compute Pipeline" );
			vkDestroyShaderModule( device, BufferPresentShader, nullptr );

			// deletors for the pipeline layout + pipeline
			mainDeletionQueue.push_function( [ & ] () {
				vkDestroyDescriptorSetLayout( device, BufferPresent.descriptorSetLayout, nullptr );
				vkDestroyPipelineLayout( device, BufferPresent.pipelineLayout, nullptr );
				vkDestroyPipeline( device, BufferPresent.pipeline, nullptr );
			});
		}

		// invoke() lambda
		BufferPresent.invoke = [ & ]( VkCommandBuffer cmd ) {
			BufferPresent.descriptorSet = getCurrentFrame().frameDescriptors.allocate( device, BufferPresent.descriptorSetLayout );
			{
				DescriptorWriter writer;
				writer.write_buffer( 0, GlobalUBO.buffer, sizeof( GlobalData ), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER );
				writer.write_image( 1, drawImage.imageView, defaultSamplerNearest, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE );
				writer.write_image( 2, Accumulator.imageView, defaultSamplerLinear, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER );
				writer.update_set( device, BufferPresent.descriptorSet );
			}

			vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, BufferPresent.pipeline );

			// bind the descriptor set, as just recorded
			vkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, BufferPresent.pipelineLayout, 0, 1, &BufferPresent.descriptorSet, 0, nullptr );

			// get a new wang RNG seed
			BufferPresent.pushConstants.wangSeed = genWangSeed();

			// send the current value of the push constants
			vkCmdPushConstants( cmd, BufferPresent.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( PushConstants ), &BufferPresent.pushConstants );

			// and the actual compute dispatch for the simulation agents
			vkCmdDispatch( cmd, ( drawExtent.width + 15 ) / 16, ( drawExtent.height + 15 ) / 16, 1 );
		};
	}
}

AllocatedBuffer PrometheusInstance::createBuffer ( size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage ) {
	// allocate buffer
	VkBufferCreateInfo bufferInfo = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bufferInfo.pNext = nullptr;
	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;

	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = memoryUsage;
	vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	AllocatedBuffer newBuffer;

	// allocate the buffer
	VK_CHECK( vmaCreateBuffer( allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info ) );

	return newBuffer;
}

void PrometheusInstance::destroyBuffer ( const AllocatedBuffer& buffer ) {
	vmaDestroyBuffer( allocator, buffer.buffer, buffer.allocation );
}

AllocatedImage PrometheusInstance::createImage ( VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped ) {
	AllocatedImage newImage;
	newImage.imageFormat = format;
	newImage.imageExtent = size;

	VkImageCreateInfo img_info = vkinit::image_create_info( format, usage, size );
	if ( mipmapped ) {
		img_info.mipLevels = static_cast<uint32_t>( std::floor( std::log2( std::max( size.width, size.height ) ) ) ) + 1;
	}

	// always allocate images on dedicated GPU memory
	VmaAllocationCreateInfo allocinfo = {};
	allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	allocinfo.requiredFlags = VkMemoryPropertyFlags( VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );

	// allocate and create the image
	VK_CHECK( vmaCreateImage( allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr ) );

	// if the format is a depth format, we will need to have it use the correct aspect flag
	VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
	if ( format == VK_FORMAT_D32_SFLOAT ) {
		aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
	}

	// build a image-view for the image
	VkImageViewCreateInfo view_info = vkinit::imageview_create_info( format, newImage.image, aspectFlag );
	view_info.subresourceRange.levelCount = img_info.mipLevels;

	VK_CHECK( vkCreateImageView( device, &view_info, nullptr, &newImage.imageView ) );

	return newImage;
}

AllocatedImage PrometheusInstance::createImage ( void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped ) {
	size_t dataSize = size.depth * size.width * size.height * 4;
	AllocatedBuffer uploadbuffer = createBuffer( dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU );

	// data from the void pointer, copied to the upload buffer
	memcpy( uploadbuffer.info.pMappedData, data, dataSize );

	// call to the read/write styled image creation function
	AllocatedImage new_image = createImage( size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped );

	// immediate mode submission, to copy the upload buffer to the allocated image
	immediateSubmit( [ & ] ( VkCommandBuffer cmd ) {
		vkutil::transition_image( cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );

		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;

		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = size;

		// copy the buffer into the image
		vkCmdCopyBufferToImage( cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion );

		// flagging the data as read-only, for shader reading... could just as easily do
		vkutil::transition_image( cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
	});

	// finished uploading, that data is now available
	destroyBuffer( uploadbuffer );

	return new_image;
}

void PrometheusInstance::updateImage( AllocatedImage& image, void* data, int bytesPerTexel ) {
	size_t dataSize = image.imageExtent.width * image.imageExtent.height * image.imageExtent.depth * bytesPerTexel;

	AllocatedBuffer uploadbuffer = createBuffer( dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU );

	memcpy( uploadbuffer.info.pMappedData, data, dataSize );


	immediateSubmit( [&]( VkCommandBuffer cmd ) {
		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;

		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = image.imageExtent;

		vkutil::transition_image( cmd, image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );
		vkCmdCopyBufferToImage( cmd, uploadbuffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion );
		vkutil::transition_image( cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL );
	} );

	destroyBuffer( uploadbuffer );
}

// this is a pretty specialized screenshot function, because it operates on the half floats stored in the draw image
void PrometheusInstance::screenshot() {
	std::string filenameS = std::string( timeDateString() + ".png" );
	const char* filename = filenameS.c_str();
	AllocatedImage& image = drawImage;
	VkExtent3D size{ drawExtent.width, drawExtent.height, 1 };

	size_t pixelCount = size.width * size.height;
	size_t dataSize = pixelCount * 4 * sizeof( uint16_t );

	AllocatedBuffer readbackBuffer = createBuffer( dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU );

	immediateSubmit( [&]( VkCommandBuffer cmd ) {
		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = size;

		vkCmdCopyImageToBuffer( cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer.buffer, 1, &copyRegion );
	} );

	std::jthread writeThread = std::jthread( [ & ] ( ) {
		uint16_t* src = ( uint16_t* ) readbackBuffer.info.pMappedData;
		std::vector<uint8_t> out( pixelCount * 4 );

		for ( size_t i = 0; i < pixelCount; i++ ) {
			float r = glm::unpackHalf1x16( src[ i * 4 + 0 ] );
			float g = glm::unpackHalf1x16( src[ i * 4 + 1 ] );
			float b = glm::unpackHalf1x16( src[ i * 4 + 2 ] );

			r = glm::clamp( r, 0.0f, 1.0f );
			g = glm::clamp( g, 0.0f, 1.0f );
			b = glm::clamp( b, 0.0f, 1.0f );

			out[ i * 4 + 0 ] = ( uint8_t ) ( r * 255.0f );
			out[ i * 4 + 1 ] = ( uint8_t ) ( g * 255.0f );
			out[ i * 4 + 2 ] = ( uint8_t ) ( b * 255.0f );
			out[ i * 4 + 3 ] = 255;
		}

		stbi_write_png( filename, size.width, size.height, 4, out.data(), size.width * 4 );
		destroyBuffer( readbackBuffer );
	});
}

void PrometheusInstance::destroyImage ( const AllocatedImage& img ) {
	vkDestroyImageView( device, img.imageView, nullptr );
	vmaDestroyImage( allocator, img.image, img.allocation );
}

void PrometheusInstance::initDefaultData () {

// need to do engine startup config soon
	// YAML::Node config = YAML::LoadFile( "../src/presets.yaml" );
	// size_t numEntries = config.size();
	// for ( size_t i = 0; i < numEntries; i++ ) {
		// presets.push_back( config[ i ].as< uint32_t >() );
	// }

// TEXTURES
	// 3 default textures, white, grey, black. 1 pixel each
	uint32_t white = glm::packUnorm4x8( glm::vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );
	whiteImage = createImage( ( void * ) &white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT );

	uint32_t grey = glm::packUnorm4x8(glm::vec4( 0.66f, 0.66f, 0.66f, 1 ) );
	greyImage = createImage( ( void * ) &grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT );

	uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0 ) );
	blackImage = createImage( ( void * ) &black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT );

// SAMPLER OBJECTS
	VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

	sampl.magFilter = VK_FILTER_NEAREST;
	sampl.minFilter = VK_FILTER_NEAREST;
	vkCreateSampler( device, &sampl, nullptr, &defaultSamplerNearest );

	sampl.magFilter = VK_FILTER_LINEAR;
	sampl.minFilter = VK_FILTER_LINEAR;
	// sampl.magFilter = VK_FILTER_CUBIC_EXT;
	// sampl.minFilter = VK_FILTER_CUBIC_EXT;
	vkCreateSampler( device, &sampl, nullptr, &defaultSamplerLinear );

	mainDeletionQueue.push_function([&](){
		vkDestroySampler( device, defaultSamplerNearest,nullptr );
		vkDestroySampler( device, defaultSamplerLinear,nullptr );

		destroyImage( whiteImage );
		destroyImage( greyImage );
		destroyImage( blackImage );
	});
}

void PrometheusInstance::initImgui () {
	// 1: create descriptor pool for IMGUI
	//  the size of the pool is very oversize, but it's copied from imgui demo
	//  itself.
	VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = ( uint32_t ) std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK( vkCreateDescriptorPool( device, &pool_info, nullptr, &imguiPool ) );

	// 2: initialize imgui library
	// this initializes the core structures of imgui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // optional

	ImGui::StyleColorsDark();

	// this initializes imgui for SDL
	ImGui_ImplSDL3_InitForVulkan( window );

	// this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = instance;
	init_info.PhysicalDevice = physicalDevice;
	init_info.Device = device;
	init_info.Queue = graphicsQueue;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	init_info.UseDynamicRendering = true;

	init_info.PipelineInfoMain = {};
	init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineRenderingCreateInfoKHR pipeline_rendering_info = {};
	pipeline_rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
	pipeline_rendering_info.colorAttachmentCount = 1;
	pipeline_rendering_info.pColorAttachmentFormats = &swapchainImageFormat;
	init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipeline_rendering_info;

	ImGui_ImplVulkan_Init( &init_info );

	// add the destroy the imgui created structures
	mainDeletionQueue.push_function( [ = ] () {
		ImGui_ImplVulkan_Shutdown();
		vkDestroyDescriptorPool( device, imguiPool, nullptr );

		ImGui_ImplSDL3_Shutdown();
		ImGui::DestroyContext();
	});
}

void PrometheusInstance::initPoints () {
	// doing the allocation
	static bool firstTime = true;

	if ( firstTime ) {
		firstTime = false;

		// create the buffer
		pointBuffer = createBuffer( sizeof( point ) * ( numPoints + numPointsFluid ), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU );
		SetDebugName( VK_OBJECT_TYPE_BUFFER, ( uint64_t ) pointBuffer.buffer, "Point Buffer" );

		mainDeletionQueue.push_function( [ = ] () {
			destroyBuffer( pointBuffer );
		});
	}

	// updating point values
	point* data = ( point* ) pointBuffer.allocation->GetMappedData();
	vec2 centerpoint = vec2( SimResolution.width, SimResolution.height / 2.0f ) / 2.0f;

	static thread_local std::mt19937 seedRNG( [] {
		std::random_device rd;
		std::seed_seq seq{  rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
		return std::mt19937( seq );
	} () );

	// first the neo-hookean particles
	for ( int x = 0; x < gridWidth; x++ ) {
		for ( int y = 0; y < gridHeight; y++ ) {
			// float massX = std::uniform_real_distribution< float >( 3.0f, 5.0f )( seedRNG );
			// float rotX = std::uniform_real_distribution< float >( 0.1f, 100.5f )( seedRNG );

			const int idx = x + gridWidth * y;
			data[ idx ] = point();

			data[ idx ].position = centerpoint + vec2( x - gridWidth / 2.0f, y - gridHeight / 2.0f ) * gridScalar;
			data[ idx ].velocity = vec2( 0.0f, 0.0f );
			data[ idx ].C = glm::mat2( 0.0f );
			data[ idx ].Fs = glm::mat2( 1.0f );
			data[ idx ].particleType = 0;
			data[ idx ].mass = 1.0f;
			// data[ idx ].mass = massX;
		}
	}

	// then initialize the fluid particles, offset in the buffer
	centerpoint.y += SimResolution.height / 3.0f;
	for ( int x = 0; x < gridWidth; x++ ) {
		for ( int y = 0; y < gridHeight / 2; y++ ) {
			const int idx = numPoints + x + gridWidth * y;
			data[ idx ] = point();
			data[ idx ].position = centerpoint + /* int( x / ( gridWidth / 15 ) ) * 10.0f + */ vec2( x - gridWidth / 2.0f, y - gridHeight / 2.0f ) * gridScalar * 1.5f;
			data[ idx ].velocity = vec2( 0.0f, -1.0f );
			data[ idx ].C = glm::mat2( 0.0f );
			data[ idx ].particleType = 1; // fluid
			data[ idx ].mass = 1.0f;
		}
	}

}

void PrometheusInstance::BufferClears ( VkCommandBuffer cmd ) {
	// clearing the grid can just be clears
	VkImageSubresourceRange range = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0,
		.levelCount = VK_REMAINING_MIP_LEVELS,
		.baseArrayLayer = 0,
		.layerCount = VK_REMAINING_ARRAY_LAYERS
	};
	VkClearColorValue clearColor;
	clearColor.int32[ 0 ] = 0;
	clearColor.int32[ 1 ] = 0;
	clearColor.int32[ 2 ] = 0;
	clearColor.int32[ 3 ] = 0;
	vkCmdClearColorImage( cmd, velocityXAtomic.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range );
	vkCmdClearColorImage( cmd, velocityYAtomic.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range );
	vkCmdClearColorImage( cmd, massAtomic.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range );
	VkImageMemoryBarrier2 barriers[ 3 ] = {
		makeImageBarrier( velocityXAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
		makeImageBarrier( velocityYAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT ),
		makeImageBarrier( massAtomic.image, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT )
	};
	VkDependencyInfo barrierDependency {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 3,
		.pImageMemoryBarriers = barriers
	};
	vkCmdPipelineBarrier2( cmd, &barrierDependency );
}

//==============================================================================================
// swapchain helpers
//==============================================================================================
void PrometheusInstance::resizeSwapchain () {
	// wait till the device shows as idle
	vkDeviceWaitIdle( device );

	// kill the existing swapchain
	destroySwapchain();

	// use SDL to find the new window size
	int w, h;
	SDL_GetWindowSize( window, &w, &h );
	windowExtent.width = w;
	windowExtent.height = h;

	// create the new swapchain and rearm trigger
	createSwapchain( w, h );
	resizeRequest = false;
}

void PrometheusInstance::createSwapchain ( uint32_t w, uint32_t h ) {
	vkb::SwapchainBuilder swapchainBuilder{ physicalDevice, device, surface };
	swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
	vkb::Swapchain vkbSwapchain = swapchainBuilder
		//.use_default_format_selection()
		.set_desired_format( VkSurfaceFormatKHR{ .format = swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR } )
		//use vsync present mode
		.set_desired_present_mode( VK_PRESENT_MODE_FIFO_KHR )
		.set_desired_extent( w, h )
		.add_image_usage_flags( VK_IMAGE_USAGE_TRANSFER_DST_BIT )
		.build()
		.value();

	//store swapchain and its related images
	swapchain = vkbSwapchain.swapchain;
	swapchainExtent = vkbSwapchain.extent;
	swapchainImages = vkbSwapchain.get_images().value();
	swapchainImageViews = vkbSwapchain.get_image_views().value();

	// draw image size will match the window
	VkExtent3D drawImageExtent = {
		windowExtent.width,
		windowExtent.height,
		1
	};

	// draw image config
	drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
	drawImage.imageExtent = drawImageExtent;
	VkImageUsageFlags drawImageUsages{};
	drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
	drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	VkImageCreateInfo rimg_info = vkinit::image_create_info( drawImage.imageFormat, drawImageUsages, drawImageExtent );

	// for the draw image, we want to allocate it from gpu local memory
	VmaAllocationCreateInfo rimg_allocinfo = {};
	rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags( VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	// allocate and create the color image
	vmaCreateImage( allocator, &rimg_info, &rimg_allocinfo, &drawImage.image, &drawImage.allocation, nullptr );
	// build a image-view for the draw image to use for rendering
	VkImageViewCreateInfo rview_info = vkinit::imageview_create_info( drawImage.imageFormat, drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT );
	VK_CHECK( vkCreateImageView( device, &rview_info, nullptr, &drawImage.imageView ) );

	// depth image config
	depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
	depthImage.imageExtent = drawImageExtent;
	VkImageUsageFlags depthImageUsages{};
	depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	VkImageCreateInfo dimg_info = vkinit::image_create_info( depthImage.imageFormat, depthImageUsages, drawImageExtent );
	//allocate and create the depth image
	vmaCreateImage( allocator, &dimg_info, &rimg_allocinfo, &depthImage.image, &depthImage.allocation, nullptr );
	// build a image-view for the draw image to use for rendering
	VkImageViewCreateInfo dview_info = vkinit::imageview_create_info( depthImage.imageFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT );
	VK_CHECK( vkCreateImageView( device, &dview_info, nullptr, &depthImage.imageView ) );

	// add to deletion queues
	mainDeletionQueue.push_function( [ = ] () {
		vkDestroyImageView( device, drawImage.imageView, nullptr );
		vmaDestroyImage( allocator, drawImage.image, drawImage.allocation );

		vkDestroyImageView( device, depthImage.imageView, nullptr );
		vmaDestroyImage( allocator, depthImage.image, depthImage.allocation );
	});
}

void PrometheusInstance::destroySwapchain () {
	vkDestroySwapchainKHR( device, swapchain, nullptr );
	for ( size_t i = 0; i < swapchainImageViews.size(); i++ ) {
		// we are only destroying the imageViews, the images are owned by the OS
		vkDestroyImageView( device, swapchainImageViews[ i ], nullptr );
	}
}

void PrometheusInstance::immediateSubmit( std::function< void( VkCommandBuffer cmd ) > && function ) {
	VK_CHECK( vkResetFences( device, 1, &immediateFence ) );
	VK_CHECK( vkResetCommandBuffer( immediateCommandBuffer, 0 ) );

	VkCommandBuffer cmd = immediateCommandBuffer;
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info( VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT );

	VK_CHECK( vkBeginCommandBuffer( cmd, &cmdBeginInfo ) );
	function( cmd );
	VK_CHECK( vkEndCommandBuffer( cmd ) );

	VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info( cmd );
	VkSubmitInfo2 submit = vkinit::submit_info( &cmdinfo, nullptr, nullptr );

	// submit command buffer to the queue and execute it.
	//  _renderFence will now block until the graphic commands finish execution
	VK_CHECK( vkQueueSubmit2( graphicsQueue, 1, &submit, immediateFence ) );
	VK_CHECK( vkWaitForFences( device, 1, &immediateFence, true, 9999999999 ) );
}

void PrometheusInstance::drawImgui ( VkCommandBuffer cmd, VkImageView targetImageView ) {
	VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info( targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL );
	VkRenderingInfo renderInfo = vkinit::rendering_info( swapchainExtent, &colorAttachment, nullptr );

	vkCmdBeginRendering( cmd, &renderInfo );
	ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), cmd );
	vkCmdEndRendering( cmd );
}
