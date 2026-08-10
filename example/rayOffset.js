import {
	ACESFilmicToneMapping,
	BoxGeometry,
	Mesh,
	MeshStandardMaterial,
	PerspectiveCamera,
	Scene,
	SphereGeometry,
	WebGPURenderer,
} from 'three/webgpu';
import { GradientEquirectTexture } from 'three-gpu-pathtracer';
import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';

const TARGET_SAMPLES = 256;
const useMegakernel = new URLSearchParams( location.search ).get( 'kernel' ) !== 'wavefront';
const scales = [ 1e-4, 1, 1e4 ];

const scene = new Scene();
const material = new MeshStandardMaterial( {
	color: 0xd8dde3,
	metalness: 0.85,
	roughness: 0.18,
} );

scales.forEach( ( scale, index ) => {

	const x = ( index - 1 ) * 1.4;
	const sphere = new Mesh(
		new SphereGeometry( 0.55 / scale, 64, 32 ),
		material.clone(),
	);
	sphere.scale.setScalar( scale );
	sphere.position.set( x, 0.55, 0 );

	const ground = new Mesh(
		new BoxGeometry( 1.25 / scale, 0.08 / scale, 1.25 / scale ),
		new MeshStandardMaterial( { color: 0x9da5af, roughness: 0.65 } ),
	);
	ground.scale.setScalar( scale );
	ground.position.set( x, -0.04, 0 );

	scene.add( sphere, ground );

} );

const environment = new GradientEquirectTexture();
environment.topColor.set( 0xffffff );
environment.bottomColor.set( 0x202830 );
environment.update();
scene.environment = environment;
scene.background = environment;

const camera = new PerspectiveCamera( 35, innerWidth / innerHeight, 0.1, 100 );
camera.position.set( 0, 2.1, -6.3 );
camera.lookAt( 0, 0.35, 0 );

const renderer = new WebGPURenderer( { antialias: true, trackTimestamp: false } );
renderer.init();
renderer.toneMapping = ACESFilmicToneMapping;
renderer.setSize( innerWidth, innerHeight );
renderer.setPixelRatio( 1 );
document.body.appendChild( renderer.domElement );

const pathTracer = new WebGPUPathTracer( renderer );
pathTracer.useMegakernel( useMegakernel );
pathTracer.setScene( scene, camera );

const status = document.getElementById( 'status' );
status.textContent = `${ useMegakernel ? 'MegaKernel' : 'Wavefront' }: 0 / ${ TARGET_SAMPLES } spp`;

renderer.setAnimationLoop( () => {

	if ( pathTracer.samples < TARGET_SAMPLES ) {

		pathTracer.renderSample();
		status.textContent = `${ useMegakernel ? 'MegaKernel' : 'Wavefront' }: ${ pathTracer.samples } / ${ TARGET_SAMPLES } spp`;

	} else {

		status.textContent = `${ useMegakernel ? 'MegaKernel' : 'Wavefront' }: complete (${ pathTracer.samples } spp)`;
		document.body.dataset.complete = 'true';

	}

} );
